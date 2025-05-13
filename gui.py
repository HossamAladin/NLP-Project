import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pickle
import os
import sys
from autocorrect import pipeline, preprocess

class ArabicAutocorrectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Autocorrect")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Check if vocabulary file exists
        if not os.path.exists('vocab.pkl'):
            messagebox.showwarning("Warning", "Vocabulary file not found. Some functionality may be limited.")
        
        # Configure style
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("TLabel", font=("Arial", 12), padding=5, background="#f0f0f0")
        style.configure("Header.TLabel", font=("Arial", 16, "bold"), padding=10, background="#f0f0f0")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_label = ttk.Label(main_frame, text="Arabic Text Autocorrection", style="Header.TLabel")
        header_label.pack(pady=10)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Text", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, font=("Arial", 12), height=8)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.correct_button = ttk.Button(button_frame, text="Correct Text", command=self.correct_text)
        self.correct_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear All", command=self.clear_all)
        self.clear_button.pack(side=tk.RIGHT, padx=5)
        
        # Output section
        output_frame = ttk.LabelFrame(main_frame, text="Corrected Text", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Arial", 12), height=8)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Corrections log section
        log_frame = ttk.LabelFrame(main_frame, text="Correction Details", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Arial", 12), height=6)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Configure text direction for Arabic
        self.input_text.configure(font=("Arial", 14))
        self.output_text.configure(font=("Arial", 14))
        
        # Set text direction to right-to-left for Arabic
        self.input_text.tag_configure("rtl", justify="right")
        self.output_text.tag_configure("rtl", justify="right")
        
    def correct_text(self):
        """Process the input text and display corrections"""
        input_text = self.input_text.get("1.0", tk.END).strip()
        
        if not input_text:
            messagebox.showinfo("Information", "Please enter some Arabic text to correct.")
            return
            
        # Clear previous outputs
        self.output_text.delete("1.0", tk.END)
        self.log_text.delete("1.0", tk.END)
        
        # Create a custom logger to capture the verbose output
        self.corrections = {}
        
        # Process the text
        try:
            # Custom implementation to capture corrections
            processed_text = preprocess(input_text)
            
            # Load vocabulary
            try:
                with open('vocab.pkl', 'rb') as f:
                    vocab = pickle.load(f)
            except:
                self.log_text.insert(tk.END, "Warning: No vocabulary file found. Using empty vocabulary.\n")
                vocab = {}
            
            from autocorrect import find_misspellings, generate_masked_sentences, predict
            
            misspelled_indices = find_misspellings(processed_text, vocab)
            
            if not misspelled_indices:
                self.log_text.insert(tk.END, "✅ لا توجد أخطاء إملائية واضحة.\n")
                self.output_text.insert(tk.END, processed_text)
                self.output_text.tag_add("rtl", "1.0", tk.END)
                return
                
            masked_sentences = generate_masked_sentences(processed_text, misspelled_indices)
            words = processed_text.split()
            corrections = {}
            
            for idx, masked in zip(misspelled_indices, masked_sentences):
                correction = predict(masked)
                corrections[words[idx]] = correction
                words[idx] = correction
                
            corrected_sentence = " ".join(words)
            
            # Display corrected text
            self.output_text.insert(tk.END, corrected_sentence)
            self.output_text.tag_add("rtl", "1.0", tk.END)
            
            # Display corrections log
            self.log_text.insert(tk.END, "🔍 الكلمات التي تم تصحيحها:\n")
            for original, corrected in corrections.items():
                self.log_text.insert(tk.END, f" - {original} ➤ {corrected}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            import traceback
            self.log_text.insert(tk.END, f"Error details:\n{traceback.format_exc()}")
    
    def clear_all(self):
        """Clear all text fields"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.log_text.delete("1.0", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ArabicAutocorrectApp(root)
    root.mainloop() 