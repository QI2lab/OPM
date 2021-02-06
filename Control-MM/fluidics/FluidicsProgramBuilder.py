import ipywidgets as widgets, pandas as pd
from IPython.display import display, clear_output

class FluidicProgramBuilder(object):

    def __init__(self):
        self.sources = [
         'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
         'B09', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16',
         'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24',
         'SSC', 'READOUT WASH', 'IMAGING BUFFER', 'CLEAVE', 'PAUSE', 'RUN']
        self.df_program = pd.DataFrame(columns=['round', 'source', 'volume', 'time'])

    def _create_widgets(self):
        self.source_widget = widgets.Dropdown(options=(self.sources), description='Source:')
        self.volume_widget = widgets.BoundedFloatText(value=5, min=0, max=10.0, step=0.1, description='Volume (mL):')
        self.time_widget = widgets.BoundedFloatText(value=5, min=0, max=20.0, step=0.1, description='Time (mins):')
        self.round_widget = widgets.BoundedIntText(value=1, min=1, max=16, step=1, description='Round: ')
        self.button_widget = widgets.Button(description='Add line to program', disabled=False)
        self.finish_button_widget = widgets.Button(description='Finish and save', disabled=False)
        self.filename_widget = widgets.Text(value='', placeholder='Enter filename without path', description='Filename:', disabled=False)
        self.save_button_widget = widgets.Button(description='Write program to file', disabled=False)
        self.button_widget.on_click(self._on_button_clicked)
        self.finish_button_widget.on_click(self._on_finish_button_clicked)
        self.save_button_widget.on_click(self._on_save_button_clicked)

    def _display_program(self):
        clear_output(wait=True)
        self.display_widgets()
        display(self.df_program)

    def _on_button_clicked(self, change):
        new_line = {'round': int(self.round_widget.value), 'source':self.source_widget.value,
         'volume':self.volume_widget.value,  'time':self.time_widget.value}
        self.df_program = self.df_program.append(new_line, ignore_index=True)
        self._display_program()

    def _on_finish_button_clicked(self, change):
        save_widgets = widgets.HBox((self.filename_widget, self.save_button_widget))
        display(save_widgets)

    def _on_save_button_clicked(self, change):
        df_final_program = self.df_program
        df_final_program.to_csv((self.filename_widget.value + '.csv'), index=True)

    def display_widgets(self):
        self._create_widgets()
        button_widgets = widgets.HBox((self.button_widget, self.finish_button_widget))
        all_widgets = widgets.VBox([self.source_widget, self.volume_widget, self.time_widget, self.round_widget, button_widgets])
        display(all_widgets)

    def define_program(self):
        clear_output(wait=True)
        self.display_widgets()
        display(self.df_program)