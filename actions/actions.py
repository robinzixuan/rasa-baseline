# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"
import csv
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import BotUttered, AllSlotsReset, SlotSet

from pymongo import MongoClient

import warnings
import pandas as pd
import numpy as np
import logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)









class ActionRecommand(Action):
   def name(self) -> Text:
      return "action_recommend"

   def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return []




class ActionRecommandAgain(Action):
   def name(self) -> Text:
      return "action_recommend_again"

   def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    

        
        return []


class ActionDiscount(Action):
   def name(self) -> Text:
      return "action_discount"

   def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        
                
        return []



class ActionReset(Action):

     def name(self) -> Text:
            return "action_reset"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         dispatcher.utter_message("All the slots has been reset")

         return [AllSlotsReset()]






def Actionbye(Action):
    def name(self) -> Text:
            return "action_bye"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message("See you next time")

        return [AllSlotsReset()]