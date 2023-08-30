from tkinter import ttk, Tk
from PIL import Image, ImageTk
import tkinter as tk
import copy
class AutoScroll(object):
    '''Configure the scrollbars for a widget.'''
    def __init__(self, master):
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))
        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''
        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)
        return wrapped

class ScrolledListBox(AutoScroll, tk.Canvas):
    def __init__(self, master, **kw):
        tk.Canvas.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)
        self.image_cache = {}
        self.original_images = {}
        self.data_list = []
        self.bind('<Configure>', self._update_layout)
        self.bind("<MouseWheel>", self.scroll)
        self.min_canvas_width = 220  
        self.spacing = 10  
        self.selector_color = "blue"
        self.text_color = "white"
    def _update_layout(self, event):
        canvas_width = self.winfo_width() - 8
        if canvas_width < self.min_canvas_width:
            return

        self.delete('all')
        total_height = 0
        half_canvas_width = canvas_width // 2

        for i, data in enumerate(self.data_list):
            text, image = data

            if i not in self.original_images:
                self.original_images[i] = image.copy()

            aspect_ratio = image.width / image.height
            max_width = max(1, half_canvas_width - 30)
            max_height = max_width / aspect_ratio
            image = image.resize((int(max_width), int(max_height)))
            image_tk = ImageTk.PhotoImage(image)

            y_offset = total_height + (i * self.spacing)
            total_height += max_height + self.spacing

            # Create a bounding rectangle around the entire item (image + text combo)
            bounding_rect = self.create_rectangle(0, y_offset, canvas_width, y_offset + max_height + self.spacing, fill='', outline='')
            
            
            # This ensures the bounding rectangle is always at the top layer
            self.tag_lower(bounding_rect)
            
            # Bind the clickable rectangle to the click event

            text_id = self.create_text(half_canvas_width // 2, y_offset + max_height // 2, anchor='center', text=text, fill=self.text_color, width=(half_canvas_width))
            image_id = self.create_image(half_canvas_width + (half_canvas_width // 2), y_offset + max_height // 2, anchor='center', image=image_tk)
            # Create the clickable rectangle
            clickable_rect = self.create_rectangle(0, y_offset, canvas_width, y_offset + max_height + self.spacing, fill='', outline='')
            self.tag_bind(clickable_rect, '<Button-1>', lambda event, idx=i: self._on_item_click(event, idx))

            self.image_cache[i] = (text_id, image_id, image_tk, bounding_rect, clickable_rect)

        self.config(scrollregion=self.bbox("all"))



    def _on_item_click(self, event, idx):
        if hasattr(self, 'highlighted_rect'):
            self.itemconfig(self.highlighted_rect, fill='')  # Clear the previous highlight by setting its fill to empty
        
        # Highlight the bounding rectangle associated with the clicked item
        self.highlighted_rect = self.image_cache[idx][3]
        self.itemconfig(self.highlighted_rect, fill=self.selector_color)

        self.selected_index = idx  # Store the actual data index
        print(idx)  # print the index of the clicked item

    def reset_images(self):
        for i, image in self.original_images.items():
            self.data_list[i][1] = image
        self._update_layout(None)

    def insert_data(self, data_list):
        self.data_list = data_list
        self.original_images.clear()
        self._update_layout(None)

    def add_item(self, text, image):
        """Add a new item (text and image) to the list and update the display."""
        self.data_list.append([text, image])
        self._update_layout(None)
    def scroll(self, event):
        self.yview_scroll(-1*(event.delta//120), "units")

    def delete_by_id(self, idx):
        """Delete an item by its index."""
        if idx < 0 or idx >= len(self.data_list):
            raise ValueError("Index out of range.")
        
        # Remove the item from data_list
        del self.data_list[idx]
        
        # Remove the item from original_images, image_cache, and the canvas
        if idx in self.original_images:
            del self.original_images[idx]
        if idx in self.image_cache:
            text_id, image_id, image_tk, bounding_rect, clickable_rect = self.image_cache[idx]
            self.delete(text_id)
            self.delete(image_id)
            self.delete(bounding_rect)
            self.delete(clickable_rect)
            del self.image_cache[idx]
        
        # Adjust the keys in original_images and image_cache
        keys_original = sorted(list(self.original_images.keys()))
        keys_cache = sorted(list(self.image_cache.keys()))
        
        for key in keys_original:
            if key > idx:
                self.original_images[key-1] = self.original_images[key]
                del self.original_images[key]
        
        for key in keys_cache:
            if key > idx:
                self.image_cache[key-1] = self.image_cache[key]
                del self.image_cache[key]
        
        # Redraw the items
        self._update_layout(None)

    def delete_all(self):
        """Delete all items from the list and update the display."""
        self.data_list.clear()  # Clear the data list
        self.original_images.clear()  # Clear the original images dictionary
        self.image_cache.clear()  # Clear the image cache dictionary
        self.delete('all')  # Remove all canvas items
    def delete_selected(self):
        if hasattr(self, 'selected_index'):
            self.delete_by_id(self.selected_index)
            delattr(self, 'selected_index')

    def get_selected_id(self):
        """Retrieve the ID of the currently selected item."""
        if hasattr(self, 'selected_index'):
            return self.selected_index
        else:
            return None  # No item is currently selected
class ScrolledImageList(AutoScroll, tk.Canvas):
    def __init__(self, master, **kw):
        tk.Canvas.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)
        self.image_cache = {}
        self.original_images = {}
        self.data_list = []
        self.bind('<Configure>', self._update_layout)
        self.min_canvas_width = 220
        self.spacing = 10
        self.selector_color = "blue"
    
    def _update_layout(self, event):
        canvas_width = self.winfo_width()
        if canvas_width < self.min_canvas_width:
            return

        self.delete('all')
        total_height = 0
        half_canvas_width = canvas_width // 2

        for i, data in enumerate(self.data_list):
            left_image, right_image = data

            left_aspect_ratio = left_image.width / left_image.height
            right_aspect_ratio = right_image.width / right_image.height

            max_width = max(1, half_canvas_width - 30)

            left_max_height = max_width / left_aspect_ratio
            left_image = left_image.resize((int(max_width), int(left_max_height)))
            left_image_tk = ImageTk.PhotoImage(left_image)

            right_max_height = max_width / right_aspect_ratio
            right_image = right_image.resize((int(max_width), int(right_max_height)))
            right_image_tk = ImageTk.PhotoImage(right_image)

            max_height = max(left_max_height, right_max_height)

            y_offset = total_height + (i * self.spacing)
            total_height += max_height + self.spacing

            # Create a bounding rectangle around the entire item
            bounding_rect = self.create_rectangle(0, y_offset, canvas_width, y_offset + max_height + self.spacing, fill='', outline='')

            # This ensures the bounding rectangle is always at the top layer
            self.tag_lower(bounding_rect)

            # Bind the clickable rectangle to the click event
            left_image_id = self.create_image(half_canvas_width // 2, y_offset + max_height // 2, anchor='center', image=left_image_tk)
            right_image_id = self.create_image(half_canvas_width + (half_canvas_width // 2), y_offset + max_height // 2, anchor='center', image=right_image_tk)

            clickable_rect = self.create_rectangle(0, y_offset, canvas_width, y_offset + max_height + self.spacing, fill='', outline='')
            self.tag_bind(clickable_rect, '<Button-1>', lambda event, idx=i: self._on_item_click(event, idx))

            self.image_cache[i] = (left_image_id, right_image_id, left_image_tk, right_image_tk, bounding_rect, clickable_rect)

        self.config(scrollregion=self.bbox("all"))

    def _on_item_click(self, event, idx):
        if hasattr(self, 'highlighted_rect'):
            self.itemconfig(self.highlighted_rect, fill='')  # Clear the previous highlight by setting its fill to empty
        
        # Highlight the bounding rectangle associated with the clicked item
        self.highlighted_rect = self.image_cache[idx][4]
        self.itemconfig(self.highlighted_rect, fill=self.selector_color)
    
    def insert_data(self, image_pair_list):
        self.data_list = image_pair_list
        self.original_images.clear()
        self._update_layout(None)
def main():
    # Create the main application window
    root = tk.Tk()
    root.title("Image List")
    
    # Create a ScrolledImageList instance
    scrolled_image_list = ScrolledImageList(root)
    scrolled_image_list.grid(row=0, column=0, sticky="nsew")
    
    # Load some sample image pairs
    left_image_paths = ["face.jpg", "1621244089249.jfif", "rick.png"]
    right_image_paths = left_image_paths[::-1]
    
    left_images = [Image.open(image_path) for image_path in left_image_paths]
    right_images = [Image.open(image_path) for image_path in right_image_paths]
    
    # Insert data into the ScrolledImageList
    image_pair_list = list(zip(left_images, right_images))
    scrolled_image_list.insert_data(image_pair_list)
    
    # Configure grid rows and columns to expand
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    # Run the tkinter main loop
    root.mainloop()
class AutoScroll_horizontal(object):
    '''Configure the scrollbar for a widget.'''
    def __init__(self, master):
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)
        self.configure(xscrollcommand=self._autoscroll(hsb))
        hsb.grid(column=0, row=1, sticky='ew')
        self.grid(column=0, row=0, sticky='ew')

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''
        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)
        return wrapped

class ScrolledListBox_horizontal(AutoScroll_horizontal, tk.Canvas):
    def __init__(self, master, **kw):
        tk.Canvas.__init__(self, master, **kw)
        AutoScroll_horizontal.__init__(self, master)
        self.image_cache = {}
        self.original_images = {}
        self.data_list = []
        self.bind('<Configure>', self._update_layout)
        self.bind("<MouseWheel>", self.scroll)
        self.min_canvas_width = 50  
        self.spacing = 10  
        self.selector_color = "blue"
        self.text_color = "white"
        self.other_widget = None
        self.selected_index = None

    def _update_layout(self, event):
        print('x')
        canvas_height = self.winfo_height() - 8
        if canvas_height < 50:  # Minimum height
            return

        self.delete('all')
        total_width = 0

        for i, data in enumerate(self.data_list):
            #text, image, embedding, we don't need embedding here
            text, image, _ = data

            if i not in self.original_images:
                self.original_images[i] = image.copy()

            aspect_ratio = image.width / image.height
            max_height = canvas_height - 30  # Reserve some space for text
            max_width = max_height * aspect_ratio
            image = image.resize((int(max_width), int(max_height)))
            image_tk = ImageTk.PhotoImage(image)

            x_offset = total_width + (i * self.spacing)
            total_width += max_width + self.spacing

            # Create a bounding rectangle around the entire item (image + text combo)
            bounding_rect = self.create_rectangle(x_offset, 0, x_offset + max_width + self.spacing, canvas_height, fill='', outline='')
            
            # Bind the clickable rectangle to the click event
            image_id = self.create_image(x_offset + max_width // 2, max_height // 2, anchor='center', image=image_tk)
            text_id = self.create_text(x_offset + max_width // 2, max_height + 10, anchor='center', text=text, fill=self.text_color)
            
            # Create the clickable rectangle
            clickable_rect = self.create_rectangle(x_offset, 0, x_offset + max_width + self.spacing, canvas_height, fill='', outline='')
            self.tag_bind(clickable_rect, '<Button-1>', lambda event, idx=i: self._on_item_click(event, idx))

            self.image_cache[i] = (text_id, image_id, image_tk, bounding_rect, clickable_rect)

        self.config(scrollregion=self.bbox("all"))

    def _on_item_click(self, event, idx):
        if hasattr(self, 'highlighted_rect'):
            self.itemconfig(self.highlighted_rect, fill='')  # Clear the previous highlight by setting its fill to empty
        if self.selected_index == idx:
            print(idx)
            return
        # Highlight the bounding rectangle associated with the clicked item
        self.highlighted_rect = self.image_cache[idx][3]
        self.itemconfig(self.highlighted_rect, fill=self.selector_color)

        self.selected_index = idx  # Store the actual data index
        print(idx)  # print the index of the clicked item

    def reset_images(self):
        for i, image in self.original_images.items():
            self.data_list[i][1] = image
        self._update_layout(None)

    def insert_data(self, data_list):
        self.data_list = copy.deepcopy(data_list)  # Make a deep copy
        self.original_images.clear()
        self._update_layout(None)

    def add_item(self, text, image, embedding):
        """Add a new item (text and image) to the list and update the display."""
        self.data_list.append([text, image, embedding])
        self._update_layout(None)

    def scroll(self, event):
        self.xview_scroll(-1*(event.delta//120), "units")

    def delete_by_id(self, idx):
        """Delete an item by its index."""
        if idx < 0 or idx >= len(self.data_list):
            raise ValueError("Index out of range.")
        
        del self.data_list[idx]
        
        if idx in self.original_images:
            del self.original_images[idx]
        if idx in self.image_cache:
            text_id, image_id, image_tk, bounding_rect, clickable_rect = self.image_cache[idx]
            self.delete(text_id)
            self.delete(image_id)
            self.delete(bounding_rect)
            self.delete(clickable_rect)
            del self.image_cache[idx]
        
        keys_original = sorted(list(self.original_images.keys()))
        keys_cache = sorted(list(self.image_cache.keys()))
        
        for key in keys_original:
            if key > idx:
                self.original_images[key-1] = self.original_images[key]
                del self.original_images[key]
        
        for key in keys_cache:
            if key > idx:
                self.image_cache[key-1] = self.image_cache[key]
                del self.image_cache[key]
        
        self._update_layout(None)

    def delete_all(self):
        """Delete all items from the list and update the display."""
        self.data_list.clear()
        self.original_images.clear()
        self.image_cache.clear()
        self.delete('all')

    def delete_selected(self):
        if hasattr(self, 'selected_index'):
            self.delete_by_id(self.selected_index)
            delattr(self, 'selected_index')

    def get_selected_id(self):
        """Retrieve the ID of the currently selected item."""
        if hasattr(self, 'selected_index'):
            return self.selected_index
        else:
            return None  # No item is currently selected
def test_horizontal():
    root = Tk()
    root.geometry("800x300")
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    canvas = ScrolledListBox_horizontal(frame, bg="black", bd=0, highlightthickness=0, relief="ridge")
    canvas.grid(row=0, column=0, sticky="nsew")  # Changed from pack to grid

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    # Sample images and text
    sample_images = [
        Image.new("RGB", (100, 100), "red"),
        Image.new("RGB", (100, 100), "green"),
        Image.new("RGB", (100, 100), "blue"),
    ]

    sample_text = ["Red", "Green", "Blue"]

    sample_data = [[text, img] for text, img in zip(sample_text, sample_images)]

    # Insert    # Insert the data into the canvas
    canvas.insert_data(sample_data)

    # Add a button to add more items
    btn_add = ttk.Button(root, text="Add Item", command=lambda: canvas.add_item("New", Image.new("RGB", (100, 100), "purple")))
    btn_add.pack()

    # Add a button to delete selected item
    btn_del = ttk.Button(root, text="Delete Selected", command=canvas.delete_selected)
    btn_del.pack()

    # Add a button to delete all items
    btn_del_all = ttk.Button(root, text="Delete All", command=canvas.delete_all)
    btn_del_all.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
    exit()
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    # Create the main application window
    root = tk.Tk()
    root.title("Image List")

    # Create a ScrolledListBox instance
    scrolled_list = ScrolledListBox(root)
    scrolled_list.grid(row=0, column=0, sticky="nsew")

    # Load some sample images
    image_paths = ["face.jpg", "1621244089249.jfif", "rick.png"]
    images = [Image.open(image_path) for image_path in image_paths]
    image_texts = ["Image 1", "Image 2", "Image 3"]

    # Insert data into the ScrolledListBox
    data_list = list(zip(image_texts, images))
    scrolled_list.insert_data(data_list)

    # Define a function to add a new item to the list
    def add_item():
        new_image = Image.open("new_image.jpg")
        new_text = "New Image"
        scrolled_list.add_item(new_text, new_image)

    # Create a button to add a new item
    add_button = tk.Button(root, text="Add Item", command=add_item)
    add_button.grid(row=1, column=0, pady=10)

    # Define a function to delete the selected item
    def delete_selected():
        scrolled_list.delete_selected()

    # Create a button to delete the selected item
    delete_button = tk.Button(root, text="Delete Selected", command=delete_selected)
    delete_button.grid(row=2, column=0, pady=10)

    # Configure grid rows and columns to expand
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Run the tkinter main loop
    root.mainloop()

