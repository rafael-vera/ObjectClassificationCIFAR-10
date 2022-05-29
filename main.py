from generate_data import generate_data
from model import DeepLearningModel
from file_paths import *

def main() -> None:
    # Uncomment the line below to generate CIFAR-10 data
    #generate_data(with_augment=False)
    
    # The model is instantiated
    model = DeepLearningModel()
    
    # The first model is shallow with Adadelta optimizer
    model.build_shallow_model('Adadelta')
    model.train_model()
    model.save_model(PATH_SHALLOW_MODEL_ADADELTA)
    
    # The second model is deep with Adadelta optimizer
    model.build_deep_model('Adadelta')
    model.train_model()
    model.save_model(PATH_DEEP_MODEL_ADADELTA)
    
    # The third model is shallow with SGD optimizer
    model.build_shallow_model('SGD')
    model.train_model()
    model.save_model(PATH_SHALLOW_MODEL_SGD)
    
    # The fourth model is deep with SGD optimizer
    model.build_deep_model('SGD')
    model.train_model()
    model.save_model(PATH_DEEP_MODEL_SGD)

if __name__ == '__main__':
    main()