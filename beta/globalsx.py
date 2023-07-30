#just a file with globals so I can make everything better
import os
os.environ['OMP_NUM_THREADS'] = '1' #sometimes speeds up things
args = {}
swapper = None
swapper_enabled = True
#[face_embeddings, chosen_face]
to_swap = []
source_face = None #if all_faces is not enabled, it wouldn't be used
current_video = 0
this_frame = None
frame_move = 0
render_queue = []