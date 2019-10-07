import recognize_api as rec
import os

images_excected = {
    "dwaine0.png":"dwaine",
    "dwaine1.png":"dwaine",
    "adrian.jpg":"adrian",
    "trisha.png":"trisha",
    "malone0.jpg":"malone",
    "patrick_bateman.jpg":"unknown",
    "emma0.jpg":"emma",
    "bill0.jpg":"bill",
    "warren0.jpg":"warren",
    "danny.jpg":"danny"
}

def simple_test():
    reader = rec.Face_reader()
    prediction  = reader.classify_image(os.path.join("images","dwaine0.png"))
    print(prediction)

def evaluate_model_simple():
    reader = rec.Face_reader()
    for image, expected in images_excected.items():
        prediction  = reader.classify_image(os.path.join("images",image))
        print("expected:", expected, "prediction:", prediction.name, "prob:", prediction.probability)

if __name__ == '__main__':
    evaluate_model_simple()

