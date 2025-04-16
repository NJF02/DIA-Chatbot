from tkinter import *
from bot_config import bot_name, get_response

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

WIDTH = 800
HEIGHT = 600

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Customer Service")
        self.window.resizable(width = False, height = False)
        self.window.configure(width = WIDTH, height = HEIGHT, bg = BG_COLOR)
        
        # Head Label
        head_label = Label(self.window, text = "Welcome", bg = BG_COLOR, fg = TEXT_COLOR, font = FONT_BOLD, pady = (0.02 * HEIGHT))
        head_label.place(relwidth = 1)
        
        # Line Divider
        line = Label(self.window, width = WIDTH, bg = BG_GRAY)
        line.place(relwidth = 1, relheight = 0.02, rely = 0.08)
        
        # Text Widget
        self.text_widget = Text(self.window, width = int(0.025 * WIDTH), height = int(0.005 * HEIGHT), 
                                bg = BG_COLOR, fg = TEXT_COLOR, font = FONT, padx = (0.001 * WIDTH), pady = (0.001 * HEIGHT))
        self.text_widget.place(relwidth = 0.98, relheight = 0.8, rely = 0.1)
        self.text_widget.configure(cursor = "arrow", state = DISABLED)
        
        # Scrollbar
        scrollbar = Scrollbar(self.window)
        scrollbar.place(relheight = 0.8, relx = 0.98, rely = 0.1)
        scrollbar.configure(command = self.text_widget.yview)
        
        # Bottom Label
        bottom_label = Label(self.window, bg = BG_GRAY, height = int(0.1 * HEIGHT))
        bottom_label.place(relwidth = 1, rely = 0.9)
        
        # Message Entry Box
        self.msg_entry = Entry(bottom_label, bg = "#2C3E50", fg = TEXT_COLOR, font = FONT)
        self.msg_entry.place(relwidth = 0.75, relheight = 0.03, relx = 0.011, rely = 0.02)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # Send Button
        send_button = Button(bottom_label, text = "Send", width = int(0.025 * WIDTH), bg = BG_GRAY, font = FONT_BOLD, 
                             command = lambda : self._on_enter_pressed(None))
        send_button.place(relwidth = 0.22, relheight = 0.03, relx = 0.77, rely = 0.02)
        
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
        
        # Sentiment analysis and output
        # sentiment_msg = f"{analyse_text(msg)}\n"
        # self.text_widget.configure(state = NORMAL)
        # self.text_widget.insert(END, sentiment_msg)
        # self.text_widget.configure(state = DISABLED)
        
        # Bot response
        bot_msg = f"{bot_name}: {get_response(msg)['response']}\n\n"
        self.text_widget.configure(state = NORMAL)
        self.text_widget.insert(END, bot_msg)
        self.text_widget.configure(state = DISABLED)
        
        # Scroll to the end
        self.text_widget.see(END)
    
if __name__ == "__main__":
    app = ChatApplication()
    app.run()