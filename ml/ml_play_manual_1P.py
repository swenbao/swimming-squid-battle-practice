import random
from pprint import pprint
import orjson
import pygame
import numpy as np

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

        if pygame.K_UP in keyboard:
            actions.append("UP")
        elif pygame.K_DOWN in keyboard:
            actions.append("DOWN")
        elif pygame.K_LEFT in keyboard:
            actions.append("LEFT")
        elif pygame.K_RIGHT in keyboard:
            actions.append("RIGHT")
        else:
            actions.append("NONE")

        # # write scene_info to a json file with indent 4 
        # with open("scene_info1P.json", "a") as f:
        #     json.dump(scene_info, f, indent=4)
        #     f.write("\n")

        return actions


    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass
