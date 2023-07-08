# Blur example filter
# author: Vladislav Janvarev

from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os

modname = os.path.basename(__file__)[:-3] # calculating modname

# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "Blur filter", # name
        "version": "1.0", # version

        "default_options": {
            "power": 30,  #
        },

        "img_processor": {
            "blur": PluginBlur
        }
    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    pass

class PluginBlur(ChainImgPlugin):
    def init_plugin(self):
        pass
    def process(self, img, params:dict):
        # params can be used to transfer some img info to next processors
        import cv2
        options = self.core.plugin_options(modname)

        ksize = (int(options["power"]), int(options["power"]))

        # Using cv2.blur() method
        image = cv2.blur(img, ksize)

        return image
