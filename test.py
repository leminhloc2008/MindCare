import tkinter as tk

import tkinter as tk

window = tk.Tk()
canvas1 = tk.Canvas(window, bg="red")
canvas1.place(x=100,y=100)

canvas2 = tk.Canvas(window, bg="blue")
canvas2.place(x=100,y=130)

canvas2.tk.call('lower', canvas2._w, None)

window.mainloop()

root.mainloop()