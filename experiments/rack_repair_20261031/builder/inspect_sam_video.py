import inspect
from transformers import Sam2VideoModel,Sam2VideoProcessor
for cls,names in [(Sam2VideoModel,['forward','propagate_in_video_iterator']),
                  (Sam2VideoProcessor,['init_video_session','add_inputs_to_inference_session','post_process_masks'])]:
    for name in names:
        item=getattr(cls,name,None)
        print(cls.__name__,name,inspect.signature(item) if item else None)
        if item:print(inspect.getdoc(item))
