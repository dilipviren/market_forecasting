import tools
import pandas as pd


class DataPrep:
    def __init__(self, data, test_size: float = 0.2, run_date: str = None):
        self.data = data
        self.test_size = test_size
        self.run_date = run_date

    def split_data(self, data):
        train_data = data[:-int(len(data) * self.test_size)]
        test_data = data[-int(len(data) * self.test_size):]
        return train_data, test_data
    
    def filter_by_date(self):
        if self.run_date:
            filtered_data = self.data[self.data['date'] <= self.run_date]
            return filtered_data
        return self.data
    
    def split_by_run_date(self):
        """
        Return (train_df, test_df).
        If run_date is not set, fall back to split_data on the full dataset.
        If run_date is set, build train as rows with date <= run_date.
        The test size (number of test timestamps) is computed as int(len(train) * test_size).
        If there are existing rows with date > run_date those are used (earliest first).
        If not enough future rows exist, generate additional future timestamps with the same interval
        (rounded to seconds, minimum 1 second) as the training data and append them to form test_df.
        """

        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])

        if not self.run_date:
            return self.split_data(df)

        run_dt = pd.to_datetime(self.run_date)
        train_df = df[df['date'] <= run_dt].sort_values('date').reset_index(drop=True)
        future_df = df[df['date'] > run_dt].sort_values('date').reset_index(drop=True)

        # Determine number of test samples based on training size
        if len(train_df) > 0:
            n_test = int(len(train_df) * self.test_size)
        else:
            # fallback: base on entire df if no training rows exist
            n_test = int(len(df) * self.test_size)

        n_test = max(0, n_test)
        if n_test == 0:
            return train_df, future_df.iloc[0:0].copy().reset_index(drop=True)

        # If enough future rows exist, take the earliest n_test of them
        if len(future_df) >= n_test:
            test_df = future_df.iloc[:n_test].reset_index(drop=True)
            return train_df, test_df

        # Need to generate additional future timestamps
        needed = n_test - len(future_df)

        # Determine interval from training data (median diff). Fallback to 1 second.
        if len(train_df) >= 2:
            diffs = train_df['date'].diff().dropna()
            freq = diffs.median()
            if pd.isnull(freq) or freq <= pd.Timedelta(0):
                freq = diffs.iloc[-1] if not diffs.empty else pd.Timedelta(seconds=1)
        else:
            freq = pd.Timedelta(seconds=1)

        # Round frequency to whole seconds (minimum 1 second) to satisfy "up to seconds" requirement
        freq_seconds = max(1, int(max(pd.Timedelta(0), freq).total_seconds()))
        freq = pd.Timedelta(seconds=freq_seconds)

        # Start generating after the latest date available (either last future or last train)
        last_known = None
        if not future_df.empty:
            last_known = future_df['date'].max()
        elif not train_df.empty:
            last_known = train_df['date'].max()
        else:
            last_known = df['date'].max()

        generated_dates = [last_known + freq * (i + 1) for i in range(needed)]

        # Build generated rows: preserve columns, set non-date columns to NaN
        gen_df = pd.DataFrame(columns=df.columns)
        if generated_dates:
            gen_df = pd.DataFrame({ 'date': generated_dates })
            for c in df.columns:
                if c != 'date':
                    gen_df[c] = pd.NA

            # Ensure column order matches original
            gen_df = gen_df[df.columns]

        test_df = pd.concat([future_df, gen_df], ignore_index=True).iloc[:n_test].reset_index(drop=True)
        return train_df, test_df