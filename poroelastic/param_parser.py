__author__ = "Alexandra Diem <alexandra@simula.no>"

from configparser import ConfigParser
import re


class ParamParser(object):

    def __init__(self, fparams):
        self.fparams = fparams
        self.sim, self.units, self.params = self.get_params()


    def get_params(self):
        """
        Reads config file provided in self.fparams.
        Parameters are stored as dictionary params['name'] = value
        """
        config = ConfigParser()
        config.optionxform = str
        config.read(self.fparams)

        # Read simulation tags section
        sim = ParamParser.get_sim_section(config)

        # Read units tags
        units = ParamParser.get_units_section(config)

        # Read parameters
        params = ParamParser.get_param_section(config, units)

        return sim, units, params


    @staticmethod
    def get_sim_section(config):
        """
        Get config file options from section containing strings.

        :param config: ConfigParser object.
        """
        section = 'Simulation'
        options = config.options(section)
        section_dict = {}
        for option in options:
            value = config.get(section, option)
            section_dict[option] = value
        return section_dict


    @staticmethod
    def get_units_section(config):
        """
        Get config file options from section containing strings.

        :param config: ConfigParser object.
        :param section: Name of the section to be read.
        """
        section = 'Units'
        options = config.options(section)
        section_dict = {}
        for option in options:
            value = config.get(section, option)
            try:
                value = float(value)
            except ValueError:
                svalue = re.split('\*|/', value)
                avalue = float(svalue[0])
                pos = len(svalue[0])
                pval = 1.0
                for val in svalue[1:]: # skip last one to avoid out of bounds error
                    if val == '2':
                        avalue = avalue * pval
                    elif val == '':
                        pass
                    else:
                        if value[pos] == '*':
                            try:
                                avalue = avalue * float(val)
                                pval = float(val)
                            except ValueError:
                                avalue = avalue * section_dict[val.strip()]
                                pval = section_dict[val.strip()]
                        elif value[pos] == '/':
                            try:
                                avalue = avalue / float(val)
                                pval = float(val)
                            except ValueError:
                                avalue = avalue / section_dict[val.strip()]
                                pval = section_dict[val.strip()]
                    pos += len(val) + 1
                value = avalue
            section_dict[option] = value
        return section_dict


    @staticmethod
    def get_param_section(config, units):
        """
        Get config file options from section containing strings.

        :param config: ConfigParser object.
        :param section: Name of the section to be read.
        """
        section = 'Parameter'
        options = config.options(section)
        section_dict = {}
        for option in options:
            value = config.get(section, option)
            try:
                value = float(value)
            except ValueError:
                svalue = re.split('\*|/', value)
                avalue = float(svalue[0])
                pos = len(svalue[0])
                pval = 1.0
                for val in svalue[1:]: # skip last one to avoid out of bounds error
                    if val == '2':
                        avalue = avalue * pval
                    elif val == '':
                        pass
                    else:
                        if value[pos] == '*':
                            try:
                                avalue = avalue * float(val)
                                pval = float(val)
                            except ValueError:
                                avalue = avalue * units[val.strip()]
                                pval = units[val.strip()]
                        elif value[pos] == '/':
                            try:
                                avalue = avalue / float(val)
                                pval = float(val)
                            except ValueError:
                                avalue = avalue / units[val.strip()]
                                pval = units[val.strip()]
                    pos += len(val) + 1
                value = avalue
            section_dict[option] = value
        return section_dict
