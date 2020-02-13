from tkinter import *
root = Tk()

e = Entry(root, width=30, borderwidth=5)
e.pack()
e.insert(0, "enter your name here..")


def my_click():
    my_Label = Label(root, text="Hello "+e.get(), fg="green")
    my_Label.pack()

my_Button = Button(root, text="Enter your name ", padx=20, pady=5, command=my_click, fg="black", bg="yellow")
my_Button.pack()

root.mainloop()