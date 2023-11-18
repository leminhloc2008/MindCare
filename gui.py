from main import *

from pathlib import Path

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, StringVar, filedialog

ASSETS_PATH = "assets"

OUTPUT_PATH = Path(__file__).parent

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def displayVideo():
    start_video(pathToVideo.get())

def StartWebcam():
    webcam_detect()

def browseFile():
    filename = filedialog.askopenfilename(title="Select a File",
                                          filetypes=(
                                          ("mp4 files", "*.mp4*"), ("avi files", "*.avi*"), ("all files", "*.*")))
    pathToVideo.set(filename)

def getStreamUrl():
    streamCam(streamingUrl.get())

def getPassUserCam():
    passCam(username.get(), password.get())

def startIpCamera():
    if not entry_3.get():
        getPassUserCam()

    else:
        getStreamUrl()

def troll():
    print(pathToVideo)

if __name__ == '__main__':

    window = Tk()

    window.geometry("1246x702")
    window.configure(bg = "#3B8275")


    canvas = Canvas(window, bg = "#3B8275", height = 702, width = 1246, bd = 0, highlightthickness = 0, relief = "ridge")

    canvas.place(x = 0, y = 0)
    canvas.create_rectangle(0.0, 0.0, 221.0, 702.0, fill="#2C695E", outline="")

    canvas.create_rectangle(221.0, 0.0, 1246.0, 702.0, fill="#3B8275", outline="")

    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    image_1 = canvas.create_image(505.0, 200.0, image=image_image_1)

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    image_2 = canvas.create_image(505.0, 520.0, image=image_image_2)

    image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
    image_3 = canvas.create_image(976.0, 360.0, image=image_image_3)

    image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
    image_5 = canvas.create_image(1121.0, 108.0, image=image_image_5)

    image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
    image_6 = canvas.create_image(638.0, 426.0, image=image_image_6)

    image_image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
    image_7 = canvas.create_image(637.0, 105.0, image=image_image_7)

    canvas.create_text(410.0, 93.0, anchor="nw", text="KET NOI VOI TELEGRAM", fill="#FFFFFF", font=("Impact", 20 * -1))

    canvas.create_text(934.0, 90.0, anchor="nw", text="Nhan dien", fill="#FFFFFF", font=("Impact", 20 * -1))

    canvas.create_text(963.0, 143.0, anchor="nw", text="Video",fill="#FFFFFF", font=("Noto Sans", 18 * -1))

    canvas.create_text(935.0, 327.0, anchor="nw", text="IP CAMERA", fill="#FFFFFF", font=("Noto Sans", 18 * -1))

    canvas.create_text(946.0, 527.0, anchor="nw", text="Webcam", fill="#FFFFFF", font=("Noto Sans", 18 * -1))

    canvas.create_text(466.0, 411.0, anchor="nw", text="Cai Dat", fill="#FFFFFF", font=("Impact", 20 * -1))

    canvas.create_text(307.0, 147.0, anchor="nw",
        text="Bước 1: Tìm @SEHC trên thanh tìm kiếm telegram.\nBước 2: Nhập tin nhắn bất kì.\nBước 3: Truy cập đường link sau: https://bit.ly/48TjbDl\nBước 4: Nhập dãy số ở mục id vào thanh dưới.",
        fill="#FFFFFF", font=("ArimoRoman Regular", 16 * -1))

    canvas.create_text(320.0, 468.0, anchor="nw", text="Access Token:", fill="#FFFFFF", font=("ArimoRoman Regular", 16 * -1))

    canvas.create_text(862.0, 191.0, anchor="nw", text="Đường dẫn video", fill="#FFFFFF", font=("ArimoRoman Regular", 16 * -1))

    canvas.create_text(831.0, 367.0, anchor="nw", text="Streaming URL", fill="#FFFFFF", font=("ArimoRoman Regular", 16 * -1))

    canvas.create_text(1044.0, 371.0, anchor="nw", text="Username", fill="#FFFFFF", font=("ArimoRoman Regular", 16 * -1))

    canvas.create_text(1042.0, 441.0, anchor="nw", text="Password", fill="#FFFFFF", font=("ArimoRoman Regular", 16 * -1))

    canvas.create_text(320.0, 505.0, anchor="nw", text="User Id:", fill="#FFFFFF", font=("ArimoRoman Regular", 16 * -1))


    entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
    entry_bg_1 = canvas.create_image(528.0, 478.0, image=entry_image_1)
    entry_1 = Entry(bd=0, bg="#A6A6A6", fg="#000716", highlightthickness=0)
    entry_1.place(x=449.0, y=466.0, width=158.0, height=22.0)

    pathToVideo = StringVar()
    entry_image_2 = PhotoImage(file=relative_to_assets("entry_2.png"))
    entry_bg_2 = canvas.create_image(926.5, 232.5, image=entry_image_2)
    entry_2 = Entry(bd=0, bg="#A6A6A6", textvariable=pathToVideo, fg="#000716", highlightthickness=0)
    entry_2.place(x=810.5, y=219.0, width=232.0, height=25.0)

    streamingUrl = StringVar()
    entry_image_3 = PhotoImage(file=relative_to_assets("entry_3.png"))
    entry_bg_3 = canvas.create_image(888.5, 416.5, image=entry_image_3)
    entry_3 = Entry(bd=0, bg="#A6A6A6", textvariable=streamingUrl, fg="#000716", highlightthickness=0)
    entry_3.place(x=810.5, y=403.0, width=156.0, height=25.0)

    username = StringVar()
    entry_image_4 = PhotoImage(file=relative_to_assets("entry_4.png"))
    entry_bg_4 = canvas.create_image(1082.5, 416.5, image=entry_image_4)
    entry_4 = Entry(bd=0, bg="#A6A6A6", textvariable=username, fg="#000716", highlightthickness=0)
    entry_4.place(x=1004.5, y=403.0, width=156.0, height=25.0)

    password= StringVar()
    entry_image_5 = PhotoImage(file=relative_to_assets("entry_5.png"))
    entry_bg_5 = canvas.create_image(1082.5,  484.5, image=entry_image_5)
    entry_5 = Entry(bd=0, bg="#A6A6A6", textvariable=password, fg="#000716", highlightthickness=0)
    entry_5.place(x=1004.5, y=471.0, width=156.0, height=25.0)

    entry_image_6 = PhotoImage(file=relative_to_assets("entry_6.png"))
    entry_bg_6 = canvas.create_image(528.0,  478.0, image=entry_image_6)
    entry_6 = Entry(
        bd=0,
        bg="#A6A6A6",
        fg="#000716",
        highlightthickness=0
    )
    entry_6.place(
        x=449.0,
        y=466.0,
        width=158.0,
        height=22.0
    )

    entry_image_7 = PhotoImage(
        file=relative_to_assets("entry_7.png"))
    entry_bg_7 = canvas.create_image(
        496.0,
        517.0,
        image=entry_image_7
    )
    entry_7 = Entry(
        bd=0,
        bg="#A6A6A6",
        fg="#000716",
        highlightthickness=0
    )
    entry_7.place(
        x=405.0,
        y=505.0,
        width=182.0,
        height=22.0
    )

    image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
    image_4 = canvas.create_image(110.0, 94.0, image=image_image_4)

    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_1 = Button(image=button_image_1, borderwidth=0, highlightthickness=0, command=troll, relief="flat")
    button_1.place(x=470.0, y=292.0, width=70.0,height=28.0)

    button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
    button_2 = Button(image=button_image_2, borderwidth=0, highlightthickness=0, command=browseFile,relief="flat")
    button_2.place(x=1071.0, y=219.0, width=86.0,height=28.0)

    button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
    button_3 = Button(image=button_image_3, borderwidth=0, highlightthickness=0, command=displayVideo, relief="flat")
    button_3.place(x=936.0, y=264.0, width=86.0, height=28.0)

    button_image_4 = PhotoImage(file=relative_to_assets("button_4.png"))
    button_4 = Button(image=button_image_4, borderwidth=0, highlightthickness=0, command=StartWebcam, relief="flat")
    button_4.place(x=934.0, y=569.0, width=86.0, height=28.0)

    button_image_5 = PhotoImage(file=relative_to_assets("button_5.png"))
    button_5 = Button(image=button_image_5, borderwidth=0, highlightthickness=0, command=startIpCamera, relief="flat")
    button_5.place(x=843.0, y=468.0, width=86.0, height=28.0)

    button_image_6 = PhotoImage(file=relative_to_assets("button_6.png"))
    button_6 = Button(image=button_image_6, borderwidth=0, highlightthickness=0, command=troll, relief="flat")
    button_6.place(x=466.0, y=604.0, width=70.0, height=28.0)

    button_image_7 = PhotoImage(file=relative_to_assets("button_7.png"))
    button_7 = Button(image=button_image_7, borderwidth=0, highlightthickness=0, command=troll,relief="flat")
    button_7.place(x=44.0, y=215.0, width=133.0, height=49.0)

    button_image_8 = PhotoImage(file=relative_to_assets("button_8.png"))
    button_8 = Button(image=button_image_8, borderwidth=0, highlightthickness=0, command=troll, relief="flat")
    button_8.place(x=44.0, y=301.0, width=133.0, height=57.0)

    button_image_9 = PhotoImage(file=relative_to_assets("button_9.png"))
    button_9 = Button(image=button_image_9, borderwidth=0, highlightthickness=0, command=troll, relief="flat")
    button_9.place(x=44.0, y=394.0, width=133.0, height=57.0
                   )
    window.resizable(False, False)
    window.mainloop()
