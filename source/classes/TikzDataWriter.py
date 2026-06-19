class TikzDataWriter:
    def __init__(self, filename):
        self.filename = filename

    def write(self, data_dict):
        """
        Write dictionary data to a semicolon-separated text file.

        Parameters
        ----------
        data_dict : dict
            Keys are column names (str),
            Values are iterables of equal length (lists, tuples, numpy arrays)
        """
        # Extract headers preserving insertion order (Python 3.7+)
        headers = list(data_dict.keys())

        # Number of rows (assumes all columns have same length)
        n_rows = len(next(iter(data_dict.values())))

        with open(self.filename, "w") as file:
            # Write header
            file.write(";".join(headers) + "\n")

            # Write rows
            for i in range(n_rows):
                row = [data_dict[h][i] for h in headers]
                line = ";".join(f"{value}" for value in row)
                file.write(line + "\n")
