import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# Too see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v0
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if (self.env.lander_a1==contact.fixtureA.body or self.env.lander_a1==contact.fixtureB.body):
            self.env.game_over_a1 = True
        if (self.env.lander_a2==contact.fixtureA.body or self.env.lander_a2==contact.fixtureB.body):
            self.env.game_over_a2 = True
    
        for i in range(2):
            if self.env.legs_a1[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs_a1[i].ground_contact = True
            if self.env.legs_a2[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs_a2[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs_a1[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs_a1[i].ground_contact = False
            if self.env.legs_a2[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs_a2[i].ground_contact = False

class LunarLanderMarl(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander_a1 = None
        self.lander_a2 = None
        self.particles = []

        self.prev_reward1 = None
        self.prev_reward2 = None

        high = np.array([np.inf]*16)  #16 states from 2 landers. useful range is -1 .. +1, but spikes can be higher 
        self.observation_space = spaces.Box(-high, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (4,)) #2 agents 2 actions each //spaces.Box(-1, +1, (2,)) 
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(16) # //spaces.Discrete(4)

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander_a1)
        self.lander_a1 = None
        self.world.DestroyBody(self.legs_a1[0])
        self.world.DestroyBody(self.legs_a1[1])

        ##second agent destroy
        self.world.DestroyBody(self.lander_a2)
        self.lander_a2 = None
        self.world.DestroyBody(self.legs_a2[0])
        self.world.DestroyBody(self.legs_a2[1])


    def _reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over_a1 = False
        self.game_over_a2 = False
        self.prev_shaping1 = None
        self.prev_shaping2 = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,) )
        chunk_x  = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y  = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i],   smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append( [p1, p2, (p2[0],H), (p1[0],H)] )

        self.moon.color1 = (0.0,0.0,0.0)
        self.moon.color2 = (0.0,0.0,0.0)

        ## AGENT 1 
        initial_y = VIEWPORT_H/SCALE
        initial_x = np.random.uniform(0.1,0.9)
        self.lander_a1 = self.world.CreateDynamicBody(
            position = (initial_x*VIEWPORT_W/SCALE, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander_a1.color1 = (0.9,0.5,0.4)
        self.lander_a1.color2 = (0.3,0.3,0.5)
        self.lander_a1.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs_a1 = []
        for i in [-1,+1]:
            leg_a1 = self.world.CreateDynamicBody(
                position = (initial_x*VIEWPORT_W/SCALE - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg_a1.ground_contact = False
            leg_a1.color1 = (0.9,0.5,0.4)
            leg_a1.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander_a1,
                bodyB=leg_a1,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg_a1.joint = self.world.CreateJoint(rjd)
            self.legs_a1.append(leg_a1)

        ## AGENT 2 
        initial_x = np.random.uniform(0.1,0.9)
        self.lander_a2 = self.world.CreateDynamicBody(
            position = (initial_x*VIEWPORT_W/SCALE, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander_a2.color1 = (0.5,0.4,0.9)
        self.lander_a2.color2 = (0.3,0.3,0.5)
        self.lander_a2.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs_a2 = []
        for i in [-1,+1]:
            leg_a2 = self.world.CreateDynamicBody(
                position = (initial_x*VIEWPORT_W/SCALE - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg_a2.ground_contact = False
            leg_a2.color1 = (0.5,0.4,0.9)
            leg_a2.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander_a2,
                bodyB=leg_a2,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg_a2.joint = self.world.CreateJoint(rjd)
            self.legs_a2.append(leg_a2)

        self.drawlist = [self.lander_a1] + self.legs_a1 + [self.lander_a2] + self.legs_a2

        return self._step(np.array([0,0,0,0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x,y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0,0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl<0):
            self.world.DestroyBody(self.particles.pop(0))

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        if self.continuous:
            self.lander_a1 = self._agent_step(self.lander_a1,[action[0],action[1]])
            self.lander_a2 = self._agent_step(self.lander_a2,[action[2],action[3]])
        else:
            self.lander_a1 = self._agent_step(self.lander_a1,action%4)
            self.lander_a2 = self._agent_step(self.lander_a2,action/4)
            #print('agent1: %d, agent2: %d'%(action/4,action%4))
        
        self.world.Step(1.0/FPS, 6*30, 2*30)
        pos1 = self.lander_a1.position
        vel1 = self.lander_a1.linearVelocity
        pos2 = self.lander_a2.position
        vel2 = self.lander_a2.linearVelocity

        state = [
            (pos1.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos1.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
            vel1.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel1.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander_a1.angle,
            20.0*self.lander_a1.angularVelocity/FPS,
            1.0 if self.legs_a1[0].ground_contact else 0.0,
            1.0 if self.legs_a1[1].ground_contact else 0.0,
            (pos2.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos2.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
            vel2.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel2.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander_a2.angle,
            20.0*self.lander_a2.angularVelocity/FPS,
            1.0 if self.legs_a2[0].ground_contact else 0.0,
            1.0 if self.legs_a2[1].ground_contact else 0.0
            ]
        assert len(state)==16

        reward = 0
        reward1 = 0
        reward2 = 0
        shaping1 = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]   # And ten points for legs contact, the idea is if you
                                                              # lose contact again after landing, you get negative reward
        shaping2 = \
            - 100*np.sqrt(state[8]*state[8] + state[9]*state[9]) \
            - 100*np.sqrt(state[10]*state[10] + state[11]*state[11]) \
            - 100*abs(state[12]) + 10*state[14] + 10*state[15]   # And ten points for legs contact, the idea is if you
                                                              # lose contact again after landing, you get negative reward

        if (self.prev_shaping1 is not None) and (self.prev_shaping2 is not None):
            reward1 = shaping1 - self.prev_shaping1# - self.prev_shaping
            reward2 = shaping2 - self.prev_shaping2
        self.prev_shaping1 = shaping1
        self.prev_shaping2 = shaping2

#        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
#        reward -= s_power*0.03
        if (reward1 >= 0 and reward2 >= 0) or (reward1 < 0 and reward2 < 0):
            reward = reward1 + reward2
        else:
            reward = 0
            
        done = False
        if ((self.game_over_a1 or abs(state[0]) >= 1.0) and (self.game_over_a2 or abs(state[8]) >= 1.0)):
            done   = True
            reward = -200
        if not (self.lander_a1.awake):
            reward = +100
        if not (self.lander_a2.awake):
            reward = +100
        if not (self.lander_a1.awake and self.lander_a2.awake):
            done   = True
            
        return np.array(state), reward, done, {}

    def _agent_step(self, agent, action):
        # Engines
        tip  = (math.sin(agent.angle), math.cos(agent.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (agent.position[0] + ox, agent.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( ox*MAIN_ENGINE_POWER*m_power,  oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            agent.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (agent.position[0] + ox - tip[0]*17/SCALE, agent.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( ox*SIDE_ENGINE_POWER*s_power,  oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            agent.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        return agent

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0,0,0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)], color=(0.8,0.8,0) )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class LunarLanderContinuousMarl(LunarLanderMarl):
    continuous = True
