# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:18:54 2017

@author: jose
"""


import sys
import os
sys.path.append(os.path.abspath("/home/jose/cligs/"))
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/"))
sys.path.append(os.path.abspath("/home/jose/Dropbox/MTB/investigacion/mytoolbox/reading_robot/"))
from toolbox.extract import read_tei, get_metadata, prepare_subcorpus


from mytoolbox.reading_robot import use_supervised, load_data, use_ml_rules

definitions_rules = [
["narrator", {"dijo": "third-person"}, {"dije": "first-person", "dijiste": "second-person"}, 0.9],
["protagonist-age",{"crecido": "adult"},{"niño": "child", "anciano" : "mature", "joven": "young"}, 0.25],
["protagonist-gender", {"él": "male"}, {"ella": "female"}, 0.5],
["protagonist-profession", {"señoritos":"lord"}, {"siervos":"servant", "soldados":"military personnel", "artistas":"artist", "párrocos":"religious", "médicos":"doctor", "dueños":"business owner", "vendedores":"seller", "estudiantes":"student", "periodistas":"journalist", "méndigos":"beggar", "científicos":"researcher", "profesores":"teacher", "trabajadores":"worker", "esposas":"housewife", "parlamentarios":"politician", "marineros":"seaman", "funcionarios":"public servant", }, 0.9],
["protagonist-social-level" , {"puesto": "medium"}, {"riqueza": "high", "miseria" : "low"}, 0.66],
["representation", {"real": "realistic"}, {"imaginario": "non-realistic"}, 1],
["setting-type",{"pueblos": "rural"},{"capital": "big-city", "provincia": "small-city", "barco":"boat"}, 0.6],
["setting-continent", {"españa": "Europe"}, {"méxico": "America", "palestina":"Asia", "áfrica": "Africa"}, 0.3],
["setting-continent_binary", {"españa": "Europe"}, {"méxico": "not_Europe"}, 0.3],
["setting-country", {"españa": "Spain"}, {"francia": "France", "italia": "Italy", "atarjea": "Atarjea", "israel": "Israel",}, 0.9],
["setting-country_binary", {"españa": "Spain"}, {"francia": "not_Spain"}, 0.9],
["setting-name", {"madrid": "Madrid"},{"marineda": "Marineda","valencia": "Valencia","parís": "París","bilbao": "Bilbao","brandeso": "Brandeso","cartagena": "Cartagena","jerusalén": "Jerusalén","oleza": "Oleza","ulloa": "Pazos de Ulloa","roma": "Roma","toledo": "Toledo","vetusta": "Vetusta"}, 0.5],
["time-period",{"actualidad": "contemporary"},{"renacimiento": "modern_times","romanos": "antiquity"}, 0],
["type-end", {"triste": "negative"}, {"feliz": "positive", "pasivo": "neutral"}, 0.55],
 ]

df_rules_results = use_ml_rules.use_ml_rules_all(
    wdir = "/home/jose/Dropbox/Doktorarbeit/publications/2018 DH/data/",
    freq_table = "freq_table_raw.csv", 
    sep = ",", 
    definitions_rules = definitions_rules,
    min_MFF = 0,
    max_MFF = 5000,
    )
