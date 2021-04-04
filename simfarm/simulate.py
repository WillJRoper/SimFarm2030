# from model_daily_3d import cultivarModel
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import pystan
import os
import pickle
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.button import Button

from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')


class Layout(GridLayout):

    def __init__(self, **kwargs):
        super(Layout, self).__init__(**kwargs)

        self.cols = 2
        self.model_path = '../cultivar_models/'
        self.model_files = os.listdir(self.model_path)

        for mfile in iter(self.model_files):
            print(mfile)
            if mfile == '.DS_Store':
                self.model_files.remove(mfile)

        self.cultivar_dict = {}

        self.inside = GridLayout()
        self.inside.cols = 1

        self.inside.dropdown = DropDown()

        for model_file in self.model_files:

            print(model_file)

            if os.path.isdir(model_file) or model_file == '.DS_Store':
                print(model_file)
                continue

            # Get the name of this cultivar from the model file name
            cultivar_name = model_file.split('_')[0]

            with open(self.model_path + model_file, 'rb') as pfile1:
                self.cultivar_dict[cultivar_name] = pickle.load(pfile1)

            btn = Button(text=cultivar_name, size_hint_y=None, height=44)

            btn.bind(on_release=self.change_model)

            self.inside.dropdown.add_widget(btn)

        # Get the name of this cultivar from the model file name
        cultivar_name = self.model_files[0].split('_')[0]
        self.current_model = self.cultivar_dict[cultivar_name]

        self.current_cultivar = cultivar_name

        # create a big main button
        self.inside.dropbutton = Button(text='Cultivar Models', size_hint=(None, None))
        self.inside.dropbutton.bind(on_release=self.inside.dropdown.open)
        self.inside.add_widget(self.inside.dropbutton)

        self.meant_label = Label(text=r'Mean Temperature Modifier: ')
        self.inside.add_widget(self.meant_label)
        self.mean_t_slider = Slider(min=-10, max=10, value=0, value_track=True, value_track_color=[1, 0, 0, 1])
        self.inside.add_widget(self.mean_t_slider)
        self.meant_label.text = r'Mean Temperature Modifier: ' + str(round(self.mean_t_slider.value, 2))
        self.mean_t_slider.bind(value=self.onSliderValueChange)

        self.meanp_label = Label(text=r'Mean Precipitation Modifier: ')
        self.inside.add_widget(self.meanp_label)
        self.mean_p_slider = Slider(min=-100, max=100, value=0, value_track=True, value_track_color=[1, 0, 0, 1])
        self.inside.add_widget(self.mean_p_slider)
        self.meanp_label.text = r'Mean Precipitation Modifier: ' + str(round(self.mean_p_slider.value, 2))
        self.mean_p_slider.bind(value=self.onSliderValueChange)

        self.t_label = Label(text=r'Temperature Modifier: ')
        self.inside.add_widget(self.t_label)
        self.t_slider = Slider(min=-10, max=10, value=0, value_track=True, value_track_color=[1, 0, 0, 1])
        self.inside.add_widget(self.t_slider)
        self.t_label.text = r'Temperature Modifier: ' + str(round(self.t_slider.value, 2))
        self.t_slider.bind(value=self.onSliderValueChange)

        self.p_label = Label(text=r'Precipitation Modifier: ')
        self.inside.add_widget(self.p_label)
        self.p_slider = Slider(min=-100, max=100, value=0, value_track=True, value_track_color=[1, 0, 0, 1])
        self.inside.add_widget(self.p_slider)
        self.p_label.text = r'Precipitation Modifier: ' + str(round(self.p_slider.value, 2))
        self.p_slider.bind(value=self.onSliderValueChange)

        self.year_label = Label(text='Year: ')
        self.inside.add_widget(self.year_label)
        self.year_slider = Slider(min=2002, max=2017, value=2002, step=1, value_track=True, value_track_color=[1, 0, 0, 1])
        self.inside.add_widget(self.year_slider)
        self.year_label.text = r'Year: ' + str(round(self.year_slider.value))
        self.year_slider.bind(value=self.onSliderValueChange)

        self.update_button = Button(text='Update', font_size=40)
        self.update_button.bind(on_press=self.compute_image)
        self.inside.add_widget(self.update_button)

        self.current_model.country_predict(self.year_slider.value, self.t_slider.value, self.p_slider.value,
                                           self.mean_t_slider.value, self.mean_p_slider.value, self.current_cultivar)
        self.country_image = Image(source='country_predictions/prediction_country_map_' + self.current_cultivar + '_'
                                          + str(round(self.year_slider.value)) + '_'
                                          + str(round(self.mean_t_slider.value, 3)) + '_'
                                          + str(round(self.mean_p_slider.value, 3)) + '_'
                                          + str(round(self.t_slider.value, 3)) + '_'
                                          + str(round(self.p_slider.value, 3)) + '.png')
        self.add_widget(self.country_image)

        self.add_widget(self.inside)

    def onSliderValueChange(self, instance, value):
        self.meanp_label.text = r'Mean Precipitation Modifier: ' + str(round(self.mean_p_slider.value, 2))
        self.meant_label.text = r'Mean Temperature Modifier: ' + str(round(self.mean_t_slider.value, 2))
        self.p_label.text = r'Precipitation Modifier: ' + str(round(self.p_slider.value, 2))
        self.t_label.text = r'Temperature Modifier: ' + str(round(self.t_slider.value, 2))
        self.year_label.text = r'Year: ' + str(round(self.year_slider.value))

    def compute_image(self, instance):

        print('Calculating...')

        self.current_model.country_predict(self.year_slider.value, self.t_slider.value, self.p_slider.value,
                                           self.mean_t_slider.value, self.mean_p_slider.value, self.current_cultivar)
        self.country_image.source = 'country_predictions/prediction_country_map_' \
                                    + self.current_cultivar + '_' \
                                    + str(round(self.year_slider.value)) + '_' \
                                    + str(round(self.mean_t_slider.value, 3)) + '_' \
                                    + str(round(self.mean_p_slider.value, 3)) + '_' \
                                    + str(round(self.t_slider.value, 3)) + '_' \
                                    + str(round(self.p_slider.value, 3)) + '.png'

        return

    def change_model(self, btn):

        self.inside.dropdown.select(btn.text)
        self.current_model = self.cultivar_dict[btn.text]
        self.current_cultivar = btn.text
        self.current_model.country_predict(self.year_slider.value, self.t_slider.value, self.p_slider.value,
                                           self.mean_t_slider.value, self.mean_p_slider.value, self.current_cultivar)
        self.country_image.source = 'country_predictions/prediction_country_map_' \
                                    + self.current_cultivar + '_' \
                                    + str(round(self.year_slider.value)) + '_' \
                                    + str(round(self.mean_t_slider.value, 3)) + '_' \
                                    + str(round(self.mean_p_slider.value, 3)) + '.png'

        return


class Simulator(App):

    def build(self):
        return Layout()


if __name__ == '__main__':
    Simulator().run()
