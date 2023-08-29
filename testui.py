import tkinter as tk
from tkinter import Canvas, Frame
from PIL import Image, ImageTk
import numpy as np

class ScrollableFrameWithButtons:
    def __init__(self, master):
        self.main_frame = Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=1)
        
        self.canvas = Canvas(self.main_frame)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.scrollbar = tk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar.pack(side=tk.TOP, fill=tk.X)
        
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.second_frame = Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.second_frame, anchor="nw")
        
        self.buttons = []
        
    def add_button(self, image, text):
        button = tk.Button(
            self.second_frame,
            text=text,
            image=image,
            compound=tk.TOP,
            width=50,
            height=50
        )
        button.grid(row=0, column=len(self.buttons), padx=5, pady=5)
        self.buttons.append(button)
        self.second_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Scrollable Frame with Buttons')
    
    # Create random noise image
    noise_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    noise_pil_img = Image.fromarray(noise_img, 'RGB')
    noise_tk_img = ImageTk.PhotoImage(image=noise_pil_img)
    
    scrollable_frame = ScrollableFrameWithButtons(root)
    
    def add_image_button():
        scrollable_frame.add_button(noise_tk_img, f"Text {len(scrollable_frame.buttons) + 1}")

    # Button to add new image buttons
    add_button = tk.Button(root, text="Add Image Button", command=add_image_button)
    add_button.pack()
    
    root.mainloop()
