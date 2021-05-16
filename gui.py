from tkinter import *


def changel():
    if l['text']!='Stop it!':
        l['text']='Stop it!'
    else:
        l['text']='Are you over?'


w = Tk()
l = Label(w,text = 'GUI Interface')
b = Button(w,text = 'A normal button',command = changel )
t = w.title('A normal window')
l.pack()
b.pack()
w.mainloop()