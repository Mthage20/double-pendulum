# main.py
from manim import *

class DoublePendulumScene(Scene):
    def construct(self):

        # lengths of the rods
        L1, L2 = 2, 1.5
        
        pivot = ORIGIN
        
        # Positions of the two bobs (masses)
        bob1_pos = pivot + DOWN * L1
        bob2_pos = bob1_pos + DOWN * L2
        
        # Rods created as lines connecting pivot/bobs
        rod1 = Line(pivot, bob1_pos, color=BLUE)   # first rod
        rod2 = Line(bob1_pos, bob2_pos, color=GREEN)  # second rod
        
        # bobs created as dots at their positions
        bob1 = Dot(bob1_pos, color=RED) 
        bob2 = Dot(bob2_pos, color=YELLOW)    
        
        self.add(rod1, rod2, bob1, bob2)
        
       
        self.wait(2)
