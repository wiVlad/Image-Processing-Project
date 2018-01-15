from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock


# Tutorial - https://www.youtube.com/watch?v=B79miUFD_ss 
# Docs - https://kivy.org/docs/guide/basic.html

def map(x, in_min,  in_max,  out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            vel = bounced * 1.1
            ball.velocity = vel.x, vel.y + offset


class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PongGame(Widget):
    actionQueue = ObjectProperty(None) 
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)

    def __init__(self,queue):
        print("YEAH!")
        Widget.__init__(self)
        self.actionQueue = queue

    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel

    def update(self, dt):
        self.ball.move()

        # bounce of paddles
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)

        # bounce ball off bottom or top
        if (self.ball.y < self.y) or (self.ball.top > self.top):
            self.ball.velocity_y *= -1

        # went of to a side to score point?
        if self.ball.x < self.x:
            self.player2.score += 1
            self.serve_ball(vel=(4, 0))
        if self.ball.x > self.width:
            self.player1.score += 1
            self.serve_ball(vel=(-4, 0))

        actions = self.actionQueue.get()
        x = map(actions[0],-250,250,50,550)
        y = map(actions[1],-250,250,50,550)
        print((x,y))
        #print(actions)

        if x < self.width / 3:
            self.player1.center_y = y
         #   self.player1.x = x
        if x > self.width - self.width / 3:
            self.player2.center_y = y
         #   self.player2.x = x

    def on_touch_move(self, touch):
        if touch.x < self.width / 3:
            self.player1.center_y = touch.y
            self.player1.x = touch.x
        if touch.x > self.width - self.width / 3:
            self.player2.center_y = touch.y
            self.player2.x = touch.x

class PongApp(App):
    def build(self,x):
        game = PongGame(x)
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game
