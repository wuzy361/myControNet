from canny2image_TRT import hackathon
import cv2


if __name__ == "__main__":
    path = "/home/player/pictures_croped/bird_0.jpg"
    img = cv2.imread(path)
    hk = hackathon()
    hk.initialize()

    export_onnx = False
    new_img = hk.process(img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers", 
            1, 
            256, 
            1,
            False, 
            1, 
            9, 
            2946901, 
            0.0, 
            100, 
            200,
            True,
            False)


