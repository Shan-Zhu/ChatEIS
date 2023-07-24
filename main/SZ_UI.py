# coding: utf-8

from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from SZ_EIS import *
from subprocess import Popen

class EIS_GUI:
    def __init__(self, master):
        self.master = master
        master.title("ChatEIS")

        self.file_path = StringVar()
        self.path_label = Label(master, text="EIS csv file：")
        self.path_label.grid(row=0, column=0)
        self.path_entry = Entry(master, textvariable=self.file_path)
        self.path_entry.grid(row=0, column=1)
        self.browse_button = Button(master, text="Find", command=self.browse_file)
        self.browse_button.grid(row=0, column=2)

        self.result_frame = Frame(master)
        self.result_frame.grid(row=1, column=0, columnspan=3)

        self.plot_frame = Frame(master)
        self.plot_frame.grid(row=2, column=0, columnspan=3)

        self.run_button = Button(master, text="Analysis", command=self.run_eis)
        self.run_button.grid(row=3, column=1)
        self.chatbot_button = Button(master, text="Chat", command=self.open_chatbot)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV file", "*.csv")])
        self.file_path.set(file_path)

    def run_eis(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        sample_path = self.file_path.get()
        sample_data = load_eis_data(sample_path)
        eis_type, eis_type_top3 = eis_ML_select(sample_data)

        fig, axs = plt.subplots(1, 3, figsize=(21.5, 5), facecolor='white')

        for i, et in enumerate(eis_type_top3[0][0]):
            circuit_model, initial_guess = get_model_and_guess(et)
            circuit = CustomCircuit(circuit_model, initial_guess=initial_guess)
            freq, Z = readFile(sample_path)
            freq, Z = ignoreBelowX(freq, Z)
            circuit.fit(freq, Z)

            result_text = Text(self.result_frame)
            result_text.insert(END, "EIS Type {}\n".format(i + 1))
            result_text.insert(END, "Probability {}\n".format(eis_type_top3[0][1][i]))
            result_text.insert(END, str(circuit))
            result_text.pack(side=LEFT)

            circuit_pred = circuit.predict(freq)
            plot_nyquist(Z, label='Raw Data', marker='o', markersize=5,markerfacecolor='none', markeredgecolor=(68 / 255, 114 / 255, 196 / 255), linewidth=0, ax=axs[i])
            plot_nyquist(circuit_pred, label='Fitting curve', fmt='-', color='red', linewidth=2, ax=axs[i])
            axs[i].legend()
            axs[i].set_xlabel('Z\' (ohm)', fontsize=15)
            axs[i].set_ylabel('-Z\'\' (ohm)', fontsize=15)
            axs[i].tick_params(axis='both', labelsize=15)
            plt.savefig('output.tiff', dpi=600, format='tiff')

        fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.95, wspace=0.5, hspace=0.2)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack(side=LEFT)

        self.chatbot_button.grid(row=3, column=2)

    def open_chatbot(self):
        Popen(["python", "SZ_ChatEIS.py"])

root = Tk()
eis_gui = EIS_GUI(root)
root.mainloop()
