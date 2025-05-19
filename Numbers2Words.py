"""
Author:       Alzahra Badi

Date created: Aug 2021

Languages:    English, Korean
"""
from num2words import num2words
import re, logging


class num2text():
    def __init__(self):
        self.noun_kr = ["명", "분", "마리", "권", "병", "잔", "살", "레", "개"]
        self.eng_pron = ["th", "nd", "st", "rd"]
        self.pattern = re.compile(r"^[0-9\,]*$")

    def ConvertKr(self, text):
        # self.logger = logging.getLogger(logFile)
        # self.logger.info(f': {self.__class__.__name__} Class')
        text = re.sub('(\d+(\.\d+)?)', r' \1 ', text)
        split_text = text.split()
        indexes = [i for i, x in enumerate(split_text) if (re.search(r'\d', x))]
        try:
            for i in indexes:
                if (i + 1) < len(split_text):
                    if split_text[i + 1] in self.noun_kr:
                        if split_text[i] == '1':
                            split_text[i] = "한"
                        else:
                            split_text[i] = num2words(split_text[i], ordinal=True, lang="kor").strip("번째")
                    else:

                        split_text[i] = num2words(split_text[i], lang="kor")
                else:
                    split_text[i] = num2words(split_text[i], lang="kor")
            # self.logger.info("%(funcName)s: Succefully converted Numbers to words")
        except Exception as e:
            # self.logger.warning("could not perform the number to word conversion. Removing all the numbers from the text")
            split_text = [item for item in text.split() if not item.isdigit()]
        return " ".join(split_text)

    def ConvertEng(self, text):

        # self.logger = logging.getLogger(logFile)
        # self.logger.info(f"{self.__class__.__name__} Class")

        # text=  re.sub('(\d+(\.\d+)?)', r' \1 ',text)
        split_text = text.split()
        indexes = [i for i, x in enumerate(split_text) if (re.search(r'\d', x))]
        try:
            for i in indexes:

                flag = True if (self.pattern.findall(split_text[i]) != []) else False

                if (i + 1) < len(split_text):

                    if split_text[i + 1] in self.eng_pron:
                        if not flag:
                            split_text[i] = num2words(split_text[i], ordinal=True)
                        else:
                            split_text[i] = num2words(split_text[i].replace(",", ""), ordinal=True)
                        del split_text[i + 1]
                    else:
                        if not flag:
                            split_text[i] = num2words(split_text[i])
                        else:
                            split_text[i] = num2words(split_text[i].replace(",", ""))
                else:
                    if not flag:
                        split_text[i] = num2words(split_text[i])
                    else:
                        split_text[i] = num2words(split_text[i].replace(",", ""))
            # self.logger.info("%(funcName)s: Succefully converted Numbers to words")
        except Exception as e:
            # self.logger.warning("could not perform the number to word conversion. Removing all the numbers from the text")
            split_text = [item for item in text.split() if not item.isdigit()]

        return " ".join(split_text).replace("-", " ")
