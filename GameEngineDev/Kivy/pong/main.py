from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
import multiprocessing
from queue import Queue
import math


class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball, speed):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            finalSpeed = max(min(speed/2, 1.5), 1)
            print(finalSpeed)
            vel = bounced*finalSpeed
            ball.velocity = vel.x, vel.y + offset


class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PongGame(Widget):
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    P1_paddle_position = [(0, 0), (0, 0)]
    P2_paddle_position = [(0, 0), (0, 0)]

    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel

    def updateLocation(self, player, position):
        if(player == 1):
            self.P1_paddle_position[0] = self.P1_paddle_position[1]
            self.P1_paddle_position[1] = position
        else:
            self.P2_paddle_position[0] = self.P2_paddle_position[1]
            self.P2_paddle_position[1] = position

    def getSpeed(self, pos):
        (x0, y0) = pos[0]
        (x1, y1) = pos[1]
        dx = x1-x0
        dy = y1-y0
        return math.sqrt(dx*dx+dy*dy)

    def update(self, dt):
        self.ball.move()

        # bounce of paddles
        self.player1.bounce_ball(
            self.ball, self.getSpeed(self.P1_paddle_position))
        self.player2.bounce_ball(
            self.ball, self.getSpeed(self.P2_paddle_position))

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

    def on_touch_move(self, touch):
        # Update current position of the players' paddles
        if touch.x < self.width / 3:
            self.updateLocation(1, (touch.x, touch.y))
            self.player1.center_y = touch.y
            self.player1.x = touch.x
        if touch.x > self.width - self.width / 3:
            self.updateLocation(2, (touch.x, touch.y))
            self.player2.center_y = touch.y
            self.player2.x = touch.x


class PongApp(App):
    def build(self, x):
        game = PongGame()
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game


if __name__ == '__main__':
    q = multiprocessing.Queue()
    PongApp().run(q)