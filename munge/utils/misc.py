"""Miscellaneous util functions"""
import json
import uuid
import csv

def get_app_config(config_file):
    """
    Gets the application configuration as a dict from the given file

    :param config_file: path to the configuration file
    :return: Dict containing the application configuration
    """
    return json.load(open(config_file))

def get_uuid():
    """
    Generates and returns a random GUID

    :return: Random GUID V4
    """
    return str(uuid.uuid4())

def csv2dict(csv_file):
    """
    Converts and returns a CSV file to Dict

    :param csv_file: Path to the CSV file
    :return: Dict representation of the CSV file
    """
    rows = [r for r in csv.reader(open(csv_file))]
    return {row[0]:row[1] for row in rows[1:]}
