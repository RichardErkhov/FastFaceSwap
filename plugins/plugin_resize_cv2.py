# Resize example filter
# author: Vladislav Janvarev

from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os

modname = os.path.basename(__file__)[:-3] # calculating modname

# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "Resize filter", # name
        "version": "1.0", # version

        "default_options": {
            "scale": 2.0,  #
        },

        "img_processor": {
            "resize_cv2": PluginResizeCv2 # 1 function - init, 2 - process
        }
    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    pass

class PluginResizeCv2(ChainImgPlugin):
    def init_plugin(self):
        pass

    def process(self, img, params: dict):
        # params can be used to transfer some img info to next processors
        import cv2
        options = self.core.plugin_options(modname)

        scale = options["scale"]
        # cv.INTER_CUBIC

        image = cv2.resize(img, None, fx=scale, fy=scale)

        return image
