__author__ = "Alexandra Diem <alexandra@simula.no>"

from configparser import ConfigParser
import argparse


class ParamParser(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg")
        args = parser.parse_args()
        cfgfile = args.cfg

        try:
            f = open(cfgfile, 'r')
            data = f.read()
            f.close()
        except Exception as e:
            print(e)
            import sys; sys.exit(1)

        self.config = ConfigParser()
        self.config.optionxform = str
        self.config.read(cfgfile)

        self.p = self.get_params()


    def get_params(self):
        """
        Reads config file provided in self.fparams.
        Parameters are stored as dictionary params['name'] = value
        """
        p = {}

        # Read simulation tags section
        p['Simulation'] = ParamParser.get_sim_section(self.config)

        # Read units tags
        p['Units'] = ParamParser.get_units_section(self.config)

        # Read parameters
        p['Parameter'] = ParamParser.get_param_section(self.config, p['Units'])

        # Read material
        p['Material'] = ParamParser.get_material_section(
                                                        self.config, p['Units'])

        return p


    def write_config(self, cfgfile):
        with open(cfgfile, 'w') as configfile:
            self.config.write(configfile)


    def add_data(self, section, key, value):
        self.config.set(section, key, value)


    @staticmethod
    def get_sim_section(config):
        """
        Get config file options from section containing strings.

        :param config: ConfigParser object.
        """
        section = 'Simulation'
        options = config.items(section)
        section_dict = {}
        for key, value in options:
            section_dict[key] = value
        return section_dict


    @staticmethod
    def get_units_section(config):
        """
        Get config file options from section containing strings.

        :param config: ConfigParser object.
        :param section: Name of the section to be read.
        """
        section = 'Units'
        options = config.items(section)
        section_dict = {}
        for key, value in options:
            value = eval(value, section_dict)
            section_dict[key] = value
        return section_dict


    @staticmethod
    def get_param_section(config, units):
        """
        Get config file options from section containing strings.

        :param config: ConfigParser object.
        :param section: Name of the section to be read.
        """
        section = 'Parameter'
        options = config.items(section)
        section_dict = {}
        for key, value in options:
            if "\n" in value:
                value = list(filter(None, [x.strip() for x in value.splitlines()]))
                value = [eval(val, units) for val in value]

            else:
                value = eval(value, units)
            section_dict[key] = value
        return section_dict


    @staticmethod
    def get_material_section(config, units):
        """
        Get config file options from section containing strings.

        :param config: ConfigParser object.
        """
        section = 'Material'
        options = config.items(section)
        section_dict = {}
        for key, value in options:
            value = eval(value, units)
            section_dict[key] = value
        return section_dict
