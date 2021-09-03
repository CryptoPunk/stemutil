__author__      = "Max Vohra"
__copyright__   = "Copyright 2021, Max Vohra"

import importlib.resources
import json

model_path = None
with importlib.resources.path(__name__.rsplit('.',1)[0], 'models') as stem_path:
    model_path = stem_path

def load_package_model(name):
    import keras
    params_file = stem_path / ('%s.json' % name)
    params = json.load(open(params_file))
    model_path = params_file.parent / params['model_path']
    model = keras.models.load_model(model_path)

    return model, params

    
