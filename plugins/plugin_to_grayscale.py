# To grayscale example filter
# author: Vladislav Janvarev

from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os

modname = os.path.basename(__file__)[:-3] # calculating modname

# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "Gray scale filter", # name
        "version": "1.0", # version

        "img_processor": {
            "to_grayscale": PluginGrayscale # 1 function - init, 2 - process
        }
    }
    return manifest


class PluginGrayscale(ChainImgPlugin):
    def init_plugin(self):
        pass

    def process(self, img, params: dict):
        # params can be used to transfer some img info to next processors
        import cv2
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Duplicate the grayscale channel to all three color channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image


