from tkinter import *
from bot_config import bot_name, bot_first_msg, get_response, get_summary

WIDTH = 900
HEIGHT = 600

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        modern_bg = "#1E1E2F"
        modern_input_bg = "#2C2C3E"
        modern_button_bg = "#3A3A5C"
        modern_text_color = "#EAEAEA"
        accent_color = "#8FBCFF"
        font_header = ("Segoe UI Bold", 14)
        font_main = ("Segoe UI", 12)
        font_bold = ("Segoe UI Semibold", 13)
        
        # Pop-up window
        self.window.title("Bot Café")
        self.window.resizable(width = False, height = False)
        self.window.configure(width = WIDTH, height = HEIGHT, bg = modern_bg)
        
        # Header Label
        head_label = Label(self.window, text = "☕ Welcome to Bot Café", 
                           bg = modern_bg, fg = accent_color, font = font_header, pady = (0.025 * HEIGHT))
        head_label.place(relwidth = 1)
        
        # Divider Line
        line = Frame(self.window, bg = accent_color, height = 2)
        line.place(relwidth = 0.94, relx = 0.03, rely = 0.09)
        
        # Text Widget
        self.text_widget = Text(self.window, bg = modern_bg, fg = modern_text_color, font = font_main, wrap = "word", 
                                padx = (0.016 * WIDTH), pady = (0.02 * HEIGHT), bd = 0, relief = "flat")
        self.text_widget.place(relwidth = 0.94, relheight = 0.72, relx = 0.03, rely = 0.11)
        # First message containing some instructions
        self.text_widget.insert(END, f"{bot_name}: {bot_first_msg}\n\n")
        self.text_widget.configure(cursor = "arrow", state = DISABLED)
        
        # Scrollbar
        scrollbar = Scrollbar(self.text_widget)
        self.text_widget.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.text_widget.yview)
        scrollbar.pack(side = RIGHT, fill = Y)
        
        # Bottom Frame
        bottom_frame = Frame(self.window, bg = modern_bg)
        bottom_frame.place(relwidth = 1, relheight = 0.15, rely = 0.85)
        
        # Message Entry Box
        self.msg_entry = Entry(bottom_frame, bg = modern_input_bg, fg = modern_text_color, 
                               font = font_main, relief = "flat", insertbackground = accent_color)
        self.msg_entry.place(relwidth = 0.74, relheight = 0.5, relx = 0.03, rely = 0.2)
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        self.msg_entry.focus()

        # Send Button
        send_button = Button(bottom_frame, text = "Send", 
                             bg = modern_button_bg, fg = modern_text_color, font = font_bold, bd = 0, relief = "flat", 
                             activebackground = accent_color, activeforeground = modern_bg, 
                             command = lambda: self._on_enter_pressed(None))
        send_button.place(relwidth = 0.2, relheight = 0.5, relx = 0.77, rely = 0.2)
        
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        
    def _insert_message(self, msg, sender):
        # If no text in the entry box
        if not msg:
            return
        
        # User input
        self.msg_entry.delete(0, END)
        user_msg = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state = NORMAL)
        self.text_widget.insert(END, user_msg)
        self.text_widget.configure(state = DISABLED)
        
        # Bot response
        result = get_response(msg)
        response = result["response"]
        bot_msg = f"{bot_name}: {response}\n\n"
        self.text_widget.configure(state = NORMAL)
        self.text_widget.insert(END, bot_msg)
        self.text_widget.configure(state = DISABLED)
        
        # If user says goodbye, provide summary and stop receiving inputs
        if(result["tags"]["tag"] == "goodbye"):
            summary = get_summary()
            bot_msg = "\n\n Summary: \n\n" + summary
            self.text_widget.configure(state = NORMAL)
            self.text_widget.insert(END, bot_msg)
            self.text_widget.configure(state = DISABLED)
            self.msg_entry.configure(state = DISABLED)
        
        # Scroll to the end
        self.text_widget.see(END)
    
if __name__ == "__main__":
    app = ChatApplication()
    app.run()