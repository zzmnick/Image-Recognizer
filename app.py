from random import random
from flask import Flask, render_template, url_for, flash, redirect, request
import os
import torch
from torchvision import transforms
from PIL import Image
import pickle
from threading import Thread, Lock

threadLock = 0
threadErrorCount = 0
fileUploaded = 0
UPLOAD_FOLDER = "./uploads"
app = Flask(__name__)

mutex = Lock()




@app.route("/", methods = ["GET", "POST"])
def main_page():
    if request.method == 'POST':
        global threadLock
        global fileUploaded
        if 'file' not in request.files:
            flash('No files uploaded')
        else:           
            file = request.files['file']
            filename = file.filename
            f = open(f"{UPLOAD_FOLDER}/{filename}", "w")
            f.close()
            file.save( UPLOAD_FOLDER + "/" + filename)
            fileLocation = f"{UPLOAD_FOLDER}/{filename}"
            print(f"created successfully: {fileLocation}")
            fileUploaded += 1
            return redirect(url_for('jian_ding', fileLocation = fileLocation))
    return render_template('mainPage.html')

@app.route("/jianding", methods = ["GET", "POST"])
def jian_ding():
    global fileUploaded
    if request.method == "POST":
        print("jianding")
        return redirect(url_for('main_page'))
    else:
        global threadLock
        fileLocation = request.args['fileLocation']
        if fileLocation != "dingzhenUnavailable":
            unconvertedJiandingRes = jianding(fileLocation)
            jiandingRes, prob = unconvertedJiandingRes
            img = categoryHandler(jiandingRes)
            print(f"Categorized as: {jiandingRes}")
            os.remove(UPLOAD_FOLDER + '/' + os.listdir(f"{UPLOAD_FOLDER}")[0])
            t = Thread(target = writeView(jiandingRes, prob))
            t.start()
        else:
            global threadErrorCount
            jiandingRes = "Failed to genarate result"
            prob = "N/A"
            img = "../static/rsc/8151730FDBE02F198A2EA6FD73284271.jpg"
            threadErrorCount += 1
            if threadErrorCount > 15:
                threadLock = 0
        return render_template('jianding.html', jiandingRes = jiandingRes, prob = prob, img = img, fileUploaded = open("./static/metadata").read())


def jianding(inputPath):
    model = pickle.load(open("./model.p", "rb"))
    pearlEyeCategory = ['Anime','Fat Guy','Man','Text','Food','Landscape','Woman','Ocean','Vehicle','Digital Gadget']
    torch.manual_seed(1376666)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()

    input_image = Image.open(inputPath).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    input_batch = input_batch.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities, 1)
    top_prob *= 100
    return(pearlEyeCategory[top_catid] if top_prob > 0.3 else 'Not Sure', round(top_prob.item(), 2))


def categoryHandler(jiandingRes):
    if jiandingRes == 'Anime':
        img = "../static/rsc/anime/" + os.listdir("static/rsc/anime")[imgHandler("static/rsc/anime/")]
    elif jiandingRes == 'Fat Guy':
        img = "../static/rsc/fat/" + os.listdir("static/rsc/fat")[imgHandler("static/rsc/fat/")]
    elif jiandingRes == 'Man':
        img = "../static/rsc/man/" + os.listdir("static/rsc/man")[imgHandler("static/rsc/man/")]
    elif jiandingRes == 'Food':
        img = "../static/rsc/food/" + os.listdir("static/rsc/food")[imgHandler("static/rsc/food/")]
    elif jiandingRes == 'Landscape':
        img = "../static/rsc/landscape/" + os.listdir("static/rsc/landscape")[imgHandler("static/rsc/landscape/")]
    elif jiandingRes == 'Woman':
        img = "../static/rsc/woman/" + os.listdir("static/rsc/woman")[imgHandler("static/rsc/woman/")]
    elif jiandingRes == 'Vehicle':
        img = "../static/rsc/vehicle/" + os.listdir("static/rsc/vehicle")[imgHandler("static/rsc/vehicle/")]
    elif jiandingRes == 'Digital Gadget':
        img = "../static/rsc/digital/" + os.listdir("static/rsc/digital")[imgHandler("static/rsc/digital/")]
    elif jiandingRes == 'Ocean':
        img = "../static/rsc/ocean/" + os.listdir("static/rsc/ocean")[imgHandler("static/rsc/ocean/")]
    elif jiandingRes == 'Text':
        img = "../static/rsc/text/" + os.listdir("static/rsc/text")[imgHandler("static/rsc/text/")]
    else:
        img = "../static/rsc/woman-shrugging.webp"
    return(img)

def imgHandler(imgDir):
    return(int((len(os.listdir(imgDir)) - 1) * random()))

def writeView(jiandingRes, prob):
    global fileUploaded
    mutex.acquire()
    try:
        view = int(open("./static/metadata", "r").read())
        print(f"file Read: {view}")
        view += 1
        file = open("./static/metadata", "w", encoding='utf-8')
        file.write(str(view))
        fileUploaded = view
    finally:
        mutex.release()
    


if __name__ == '__main__':
    app.run(debug=True)




