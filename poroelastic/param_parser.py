__author__ = "Alexandra Diem <alexandra@simula.no>"

from configparser import SafeConfigParser


class ParamParser(object):

    def __init__(self, fparams):
        self.fparams = fparams
        try:
            f = open(fparams, 'r')
            data = f.read()
            f.close()
        except Exception as e:
            print(e)
            import sys; sys.exit(1)
        self.sim, self.units, self.params = self.get_params()


    def get_params(self):
        """
        Reads config file provided in self.fparams.
        Parameters are stored as dictionary params['name'] = value
        """
        config = SafeConfigParser()
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
