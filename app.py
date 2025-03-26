import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from config import Config
from analysis import FileAnalyzer

class AudioApp:

    filepath: str
    analyzer: FileAnalyzer

    def __init__(self, root):
        self.root = root
        self.root.title("Analiza sygnału audio")

        window_height = Config.WINDOW_HEIGHT
        window_width = Config.WINDOW_WIDTH
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        self.root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        menu = tk.Menu(root)
        
        file_menu = tk.Menu(menu)
        file_menu.add_command(label="Ustawienia", command=self.show_settings)
        sub_menu = tk.Menu(file_menu, tearoff=0)
        sub_menu.add_command(label='Parametry ramek', command=lambda: self.save_file(clip=False))
        sub_menu.add_command(label='Parametry klipu', command=lambda: self.save_file(clip=True))
        file_menu.add_cascade(
            label="Eksportuj",
            menu=sub_menu
        )
        menu.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Wyjście", command=self.on_closing)

        # help_menu = tk.Menu(menu)
        # help_menu.add_command(label="O aplikacji", command=lambda x: print("o aplikacji"))
        # menu.add_cascade(label="Pomoc", menu=help_menu)

        root.config(menu=menu)

        self.create_widgets()
    
    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def create_widgets(self):
        
        # file
        self.file_frame = tk.Frame(self.root)
        self.file_frame.pack()

        self.open_button = tk.Button(self.file_frame, text="Wczytaj plik WAV", command=self.open_file)
        self.open_button.pack(pady=(20, 10))

        self.file_label = tk.Label(self.file_frame, text="Wybrany plik: brak")
        self.file_label.pack()

        # analysis
        self.analysis_frame = tk.Frame(self.root)
        self.analysis_frame.pack(pady=(30, 0))

        # frame analyis
        self.frame_analysis_frame = tk.Frame(self.analysis_frame, relief=tk.RIDGE, borderwidth=2, border=1)
        self.frame_analysis_frame.pack(padx=20, side=tk.LEFT, fill='y')

        self.frame_label = tk.Label(self.frame_analysis_frame, text="Analiza ramek - wykresy")
        self.frame_label.pack(pady=5, padx=10)

        self.energy_button = tk.Button(self.frame_analysis_frame, text="Energia", command=self.plot_energy, state=tk.DISABLED)
        self.energy_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')
        
        self.zcr_button = tk.Button(self.frame_analysis_frame, text="Współczynnik ZCR", command=self.plot_zcr, state=tk.DISABLED)
        self.zcr_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.volume_button = tk.Button(self.frame_analysis_frame, text="Głośność", command=self.plot_volume, state=tk.DISABLED)
        self.volume_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')
        
        self.sr_button = tk.Button(self.frame_analysis_frame, text="Cisza", command=self.plot_silence, state=tk.DISABLED)
        self.sr_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.voicing_button = tk.Button(self.frame_analysis_frame, text="Dźwięczność", command=self.plot_voicing, state=tk.DISABLED)
        self.voicing_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        # clip analysis
        self.clip_analysis_frame = tk.Frame(self.analysis_frame, relief=tk.RIDGE, borderwidth=10, border=1)
        self.clip_analysis_frame.pack(padx=20, side=tk.RIGHT, fill='y')

        self.f0_label = tk.Label(self.clip_analysis_frame, text="Ton podstawowy")
        self.f0_label.pack(pady=5, padx=10)

        self.cor_button = tk.Button(self.clip_analysis_frame, text="Autokorelacja", command=self.plot_f0_cor, state=tk.DISABLED)
        self.cor_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.amdf_button = tk.Button(self.clip_analysis_frame, text="AMDF", command=self.plot_f0_amdf, state=tk.DISABLED)
        self.amdf_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        self.clip_label = tk.Label(self.clip_analysis_frame, text="Inne analizy")
        self.clip_label.pack(pady=(15, 5), padx=10)

        self.parameters_button = tk.Button(self.clip_analysis_frame, text="Parametry klipu", command=self.show_parameters, state=tk.DISABLED)
        self.parameters_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        # self.classification_button = tk.Button(self.clip_analysis_frame, text="Klasyfikacja", command=self.plot_classification, state=tk.DISABLED)
        # self.classification_button.pack(pady=5, padx=10, side=tk.TOP, fill='x')

        # canvas

        self.canvas_label = tk.Label(root, text="Przebieg czasowy pliku:")
        self.canvas_label.pack(anchor='w', pady=(30,0), padx=20)

        self.canvas_frame = tk.Frame(root, relief='solid', border=1, background='white')
        self.canvas_frame.pack(pady=(10, 20), padx=20)

        self.fig = plt.figure(constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(pady=10, padx=10)

    
    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        self.filepath = filepath
        try:
            self.analyzer = FileAnalyzer(filepath)
        except Exception as e:
            print("Cannot load file:", e)
            self.file_label.configure(text=f"Wybrany plik: Nie udało się wczytać wybranego pliku")
            return
        self.file_label.configure(text=f"Wybrany plik: {filepath.split('/')[-1]}")
        self.filepath = filepath
        self.activate_buttons()
        self.draw_waveform()
    
    def activate_buttons(self):
        frames = [self.clip_analysis_frame, self.frame_analysis_frame]
        for frame in frames:
            for _, child in frame.children.items():
                if isinstance(child, tk.Button):
                    child.configure(state=tk.NORMAL)
    
    def draw_waveform(self):
        ax = self.fig.gca()
        ax.clear()

        data, time = self.analyzer.waveform()

        ax.plot(time, data, label="Przebieg sygnału")

        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid()
        # ax.set_title("Przebieg czasowy sygnału audio")

        self.canvas.draw()

        plt.close(self.fig)


    def show_parameters(self):
        params = self.analyzer.calculate_parameters()
        
        self.params_window = tk.Toplevel(self.root, padx=40, pady=10)
        self.params_window.title("Parametry klipu")
        self.params_window.geometry(f"{Config.MENU_WIDTH}x{Config.MENU_HEIGHT}")
        self.root.eval(f'tk::PlaceWindow {str(self.params_window)} center')

        self.params_label_frame = tk.Frame(self.params_window, width=Config.MENU_WIDTH / 2)
        self.params_label_frame.pack(side=tk.LEFT, anchor='ne')

        self.params_value_frame = tk.Frame(self.params_window, width=Config.MENU_WIDTH / 2)
        self.params_value_frame.pack(side=tk.RIGHT, anchor='nw', padx=(0, 40))

        for key, value in params.items():
            label = tk.Label(self.params_label_frame, text=f"{key}:")
            label.pack(pady=5, anchor='e')

            if not isinstance(value, str):
                value = round(float(value), 3)
            param = tk.Label(self.params_value_frame, text=f"{value}")
            param.pack(pady=5, anchor='w')

        self.params_window.transient(self.root)
        self.params_window.grab_set()
    
    def show_settings(self):
        self.config_window = tk.Toplevel(self.root, padx=40, pady=10)
        self.config_window.title("Ustawienia parametrów")
        self.config_window.geometry(f"{Config.MENU_WIDTH}x{Config.MENU_HEIGHT}")
        self.root.eval(f'tk::PlaceWindow {str(self.config_window)} center')

        self.frame_size_var = tk.StringVar(value=str(int(Config.FRAME_SIZE * 1000)))
        self.min_f0_var = tk.StringVar(value=str(Config.MIN_F0))
        self.max_f0_var = tk.StringVar(value=str(Config.MAX_F0))
        self.entropy_k_var = tk.StringVar(value=str(Config.ENTROPY_K))

        tk.Label(self.config_window, text="Rozmiar ramki (ms):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.frame_size_var).pack()

        tk.Label(self.config_window, text="Min. F0 (hz):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.min_f0_var).pack()

        tk.Label(self.config_window, text="Maks. F0 (hz):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.max_f0_var).pack()

        tk.Label(self.config_window, text="Rozmiar segmentu K (w entropii energii):").pack(pady=5)
        tk.Entry(self.config_window, textvariable=self.entropy_k_var).pack()

        tk.Button(self.config_window, text="Zapisz", command=self.save_config).pack(pady=15)


    def save_config(self):
        try:
            Config.FRAME_SIZE = int(self.frame_size_var.get()) / 1000
            Config.MIN_F0 = int(self.min_f0_var.get())
            Config.MAX_F0 = int(self.max_f0_var.get())
            Config.ENTROPY_K = int(self.entropy_k_var.get())
            if self.filepath is not None:
                try:
                    self.analyzer = FileAnalyzer(self.filepath)
                except Exception as e:
                    print(e)
            tk.messagebox.showinfo("", "Parametry zostały zapisane")
            self.config_window.destroy()
        except ValueError as e:
            print(e)
            tk.messagebox.showerror("Błąd", "Wprowadzono niepoprawne wartości liczbowe")


    def plot_energy(self):
        energy, time = self.analyzer.ste()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, energy, label="Energia sygnału na poziomie ramek")
        plt.xlabel("Czas [s]")
        plt.ylabel("Energia")
        plt.title("Energia sygnału na poziomie ramek")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_volume(self):
        volume, time = self.analyzer.volume()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, volume, label="Głośność na poziomie ramek")
        plt.xlabel("Czas [s]")
        plt.ylabel("Głośność")
        plt.title("Głośność na poziomie ramek")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_zcr(self):
        zcr, time = self.analyzer.zcr()
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, zcr, label="Zero-Crossing Rate")
        plt.xlabel("Czas [s]")
        plt.ylabel("ZCR")
        plt.title("Współczynnik Zero-Crossing Rate na poziomie ramek")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_silence(self):
        silence, time = self.analyzer.silence()
        # sr = self.analyzer.silent_ratio()
        volume, _ = self.analyzer.volume()

        plt.figure(figsize=(10, 4))

        # to create the legend
        plt.axvspan(0, 0, color='red', alpha=0.4, label='Cisza')

        # silence segments
        delta = time[1] - time[0]
        for t, label in zip(time, silence):
            if label == 1:
                t_start = t
                t_end = t_start + delta
                plt.axvspan(t_start, t_end, color='red', alpha=0.4)

        plt.plot(time, volume, label="Głośność", alpha=0.8)

        plt.xlabel('Czas [s]')
        plt.ylabel('Głośność')
        plt.title('Cisza na poziomie ramek')
        plt.xlim(time[0], time[-1] + delta)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()

    def plot_f0_cor(self):
        f0_corr, time_corr = self.analyzer.f0(method='cor')
        mean, median, mode = self.analyzer.get_stats(f0_corr)

        plt.figure(figsize=(10, 4))
        plt.plot(time_corr, f0_corr)
        plt.xlabel("Czas [s]")
        plt.ylabel("f0 [Hz]")
        plt.suptitle("Częstotliwość tonu podstawowego - Autokorelacja")
        plt.title(f"Średnia: {mean:.1f}, Mediana: {median:.1f}, Moda: {mode:.1f}")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_f0_amdf(self):
        f0_amdf, time_amdf = self.analyzer.f0(method='amdf')
        mean, median, mode = self.analyzer.get_stats(f0_amdf)

        plt.figure(figsize=(10, 4))
        plt.plot(time_amdf, f0_amdf)
        plt.xlabel("Czas [s]")
        plt.ylabel("f0 [Hz]")
        plt.suptitle("Częstotliwość tonu podstawowego - AMDF")
        plt.title(f"Średnia: {mean:.1f}, Mediana: {median:.1f}, Moda: {mode:.1f}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_voicing(self):
        
        labels, time = self.analyzer.voicing()
        volume, _ = self.analyzer.volume()

        color_map = {
            'voiced': 'green',
            'unvoiced': 'red',
            'ambiguous': 'gray'
        }

        plt.figure(figsize=(10, 4))

        # to create the table
        for label, color in color_map.items():
            plt.axvspan(0, 0, color=color, alpha=0.4, label=label)

        # (un)voiced segments
        delta = time[1] - time[0]
        for t, label in zip(time, labels):
            if label == 'ambiguous':
                continue
            t_start = t
            t_end = t_start + delta
            plt.axvspan(t_start, t_end, color=color_map.get(label), alpha=0.4)

        plt.plot(time, volume, label="volume", alpha=0.8)

        plt.xlabel('Czas [s]')
        plt.ylabel('Głośność')
        plt.title('Podział na fragmenty dźwięczne i bezdźwięczne')
        plt.xlim(time[0], time[-1] + delta)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_file(self, clip=False):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save file as"
        )

        if file_path:
            dataframe = self.prepare_params(clip=clip)
            dataframe.to_csv(file_path)

    def prepare_params(self, clip=False):
        if clip:
            params = self.analyzer.calculate_parameters()
            return pd.DataFrame(params, index=[0])
        else:
            params = {}
            ste, params['time'] = self.analyzer.ste()
            params['ste'] = ste
            params['zcr'], _ = self.analyzer.zcr()
            params['volume'], _ = self.analyzer.volume()
            silence, _ = self.analyzer.silence()
            params['silence'] = silence.astype(int)
            params['voicing'], _ = self.analyzer.voicing()
            params['f0'], _ = self.analyzer.f0(method='cor')
            return pd.DataFrame(params)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    app.root.mainloop()
