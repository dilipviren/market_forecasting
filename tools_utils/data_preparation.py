import tools_utils.runtime_tools as runtime_tools
import pandas as pd
from datetime import datetime

config = runtime_tools.GetConfig().get_config()['columns']


class DataPrep:
    def __init__(self, test_size: float = 0.2, run_date: str = '2024-06-01'):
        self.test_size = test_size
        self.run_date = pd.to_datetime(run_date) if run_date else None

    def convert_timestamp(self, data: pd.DataFrame, time_col: str=config['date_col']) -> pd.DataFrame:
        df = data.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        return df
    
    def split_train_test(self, data: pd.DataFrame):
        df = data.copy().sort_values(by=config['date_col']).reset_index(drop=True)
        test_len = int(self.test_size * len(df))
        # print(self.run_date)
        if df[config['date_col']].dtype != 'datetime64[ns]':
            df[config['date_col']] = [pd.to_datetime(d) for d in df[config['date_col']]]
        # print(df[config['date_col']].dtype)

        if self.run_date is not None:
            if len(df[df[config['date_col']]>=self.run_date])>test_len:
                print('sufficient data exists past run date, splitting data.')
                split_index = int(len(df) - test_len)
                train_data = df.iloc[:split_index]
                test_data = df.iloc[split_index:]
                return train_data, test_data
            else:
                print('insufficient data exists past run date, generating future timestamps.')
                train_data = df[df[config['date_col']]<self.run_date]
                test_dict = {}
                for col in train_data.columns:
                    test_dict[col] = [0] * int(test_len)
                test_data = pd.DataFrame(test_dict)
                test_data[config['date_col']] = pd.date_range(start=self.run_date, periods=int(test_len), freq='D')[:]
                return train_data, test_data
        else:
            split_index = int(len(df) - test_len)
            train_data = df.iloc[:split_index]
            test_data = df.iloc[split_index:]
            return train_data, test_data

    
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