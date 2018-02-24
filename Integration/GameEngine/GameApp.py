from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
import math
from queue import Queue
import numpy as np

from kivy.graphics import Rectangle, Color, Canvas, Ellipse, Line



# Tutorial - https://www.youtube.com/watch?v=B79miUFD_ss 
# Docs - https://kivy.org/docs/guide/basic.html


NUM_OF_TRAIL = 20

def map(x, in_min,  in_max,  out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            # finalSpeed = max(min(speed/2, 1.5), 1)
            # print(finalSpeed)
            vel = bounced #*finalSpeed
            ball.velocity = vel.x, vel.y + offset



class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    powerUp = ObjectProperty(None)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos
    
    def reverseVelocity(self):
        vx, vy = self.velocity
        self.velocity = Vector((-vx, -vy))
        with self.canvas:
            Color(0, 1, 1, 0.2)
            d = 20
            Ellipse(pos=(self.pos), size=(d, d))

    def speedUp(self):
        vx, vy = self.velocity
        print(self.velocity)
        self.velocity = Vector((vx*1.6, vy*1.6))
        with self.canvas:
            Color(1, 0, 1, 0.2)
            d = 20
            Ellipse(pos=(self.pos), size=(d, d))

class PongObstacle(Widget):

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            vel = bounced
            ball.velocity = vel.x, vel.y + offset


class PongSpeeder(Widget):

    def pass_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            newVel = Vector(1.05*vx, 1.05*vy)
            ball.velocity = newVel


class PongGame(Widget):
    actionQueue = ObjectProperty(None)
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    t = NumericProperty(0)
    movesQueue = ObjectProperty(None)
    oldBalls = ObjectProperty(None)

    def __init__(self, queue):
        print("YEAH!")
        Widget.__init__(self)
        self.actionQueue = queue
        self.movesQueue = np.array([(0, 0) for x in range(NUM_OF_TRAIL)])
        self.oldBalls = [Ellipse(pos=(0, 0), size=(20, 20))
                         for x in range(NUM_OF_TRAIL)]

    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel

    def drawTrail(self):
        x = round(self.ball.x)
        y = round(self.ball.y)
        # print(movesQueue)
        # print(x, y)

        # Insert new ball location
        currentLoc = self.t % NUM_OF_TRAIL
        self.movesQueue[currentLoc] = (x, y)  # self.pos

        # Remove old trail
        for i in range(NUM_OF_TRAIL):
            with self.canvas:
                old = self.oldBalls[i]
                self.canvas.remove(old)

        #Draw new trail
        for i in range(NUM_OF_TRAIL):
            with self.canvas:
                Color(1, 0, 1, (1.0/NUM_OF_TRAIL)*i)
                d = (70/NUM_OF_TRAIL)*i
                self.oldBalls[i] = Ellipse(
                    pos=(self.movesQueue[(currentLoc+i) % NUM_OF_TRAIL]), size=(d, d))

    def update(self, dt):
        self.t = self.t + 1
        # print(self.t)
        # if(self.t % 2 == 0):
        #     self.drawTrail()
        self.ball.move()
        # bounce of paddles
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)

        # bounce ball off bottom or top
        if (self.ball.y < self.y) or (self.ball.top > self.top):
            self.ball.velocity_y *= -1

        # bounce ball off obstacle
        self.obs.bounce_ball(self.ball)

        # pass ball through speeder
        self.speeder.pass_ball(self.ball)

        # went of to a side to score point?
        if self.ball.x < self.x:
            self.player2.score += 3
            self.serve_ball(vel=(4, 0))
        if self.ball.x > self.width:
            self.player1.score += 3
            self.serve_ball(vel=(-4, 0))


        packet = self.actionQueue.get()
        (id,x,y) = packet
        y = int(470 - y)
        y = int(map(y,0,460,0,700))
        x = int(map(x,75,780,0,1150))

        
        #print("Game X/Y Values:")
        #print((id,x,y))
        #print(actions)

        #if x < self.width / 3:
        if(id == 1):
            self.player1.center_y = y
            self.player1.x = x + 170
        #if x > self.width - self.width / 3:
        if(id == 2):
            self.player2.center_y = y
            self.player2.x = x - 5

        if(id == 3):
            self.obs.x = x;
            self.obs.y = y

        if(id == 4):
            self.speeder.x = x
            self.speeder.y = y
        
        # Command - use points to reverse balls' velocity 
        # For player 2 (right)
        if(id == 5):
            if(self.player2.score >= 2):
                self.player2.score -= 2
                self.ball.reverseVelocity()

        # For player 1 (left)
        if(id == 6):
            if(self.player1.score >= 2):
                self.player1.score -= 2
                self.ball.reverseVelocity()


        # Command - use points to speed balls' velocity 
        # For player 2 (right)
        if(id == 7):
            if(self.player2.score >= 1):
                self.player2.score -= 1
                self.ball.speedUp()

        # For player 1 (left)
        if(id == 8):
            if(self.player1.score >= 1):
                self.player1.score -= 1
                self.ball.speedUp()



        # print("\n Y and X player 1:")
        # print(self.player1.center_y)
        # print(self.player1.x)
        # print("Y and X player 2:")
        # print(self.player2.center_y)
        # print(self.player2.x)


    def on_touch_move(self, touch):
        # print(touch)
        if touch.x < self.width / 3:
            self.updateLocation(1, (touch.x, touch.y))
            self.player1.center_y = touch.y
            self.player1.x = touch.x
        if touch.x > self.width - self.width / 3:
            self.updateLocation(2, (touch.x, touch.y))
            self.player2.center_y = touch.y
            self.player2.x = touch.x

class PongApp(App):
    def build(self,x):
        game = PongGame(x)
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game
