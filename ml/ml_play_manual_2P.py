import pygame
import json
import random
from pprint import pprint
import orjson


class MLPlay:
    def __init__(self,ai_name,*args,**kwargs):
        print("Initial ml script")

    def update(self, scene_info: dict, keyboard:list=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        # pprint("AI received data from game :", orjson.dumps(scene_info))
        # pprint(scene_info)
        actions = []

        if pygame.K_w in keyboard:
            actions.append("UP")
        elif pygame.K_s in keyboard:
            actions.append("DOWN")

        elif pygame.K_a in keyboard:
            actions.append("LEFT")

        elif pygame.K_d in keyboard:
            actions.append("RIGHT")
        else:
            actions.append("NONE")
            
        return actions



    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass
