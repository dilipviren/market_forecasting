import pandas as pd
import json
from pathlib import Path
import os
import sys
import yaml


class GetConfig:
    """Handles reading configuration files."""
    
    @staticmethod
    def get_config():
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / 'config' / 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        return config
    

class PortfolioReader:
    """Reads portfolio CSV files from the 'portfolio' directory."""

    @staticmethod
    def read_csv(filename: str = "stock_portfolio.csv") -> pd.DataFrame:
        csv_path = Path("portfolio") / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return pd.read_csv(csv_path)
        
    
if __name__ == "__main__":
    config = GetConfig.get_config()
    print(config)

    portfolio_df = PortfolioReader.read_csv()
    print(portfolio_df)


class PortfolioConverter:
    """Handles conversion of portfolio files between CSV and JSON formats."""

    def __init__(self, csv_path="portfolio/portfolio.csv", json_path="portfolio/portfolio.json"):
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

    def csv_to_json(self) -> None:
        """Converts the CSV portfolio to a JSON file."""
        df = pd.read_csv(self.csv_path)

        # Ensure required columns exist
        required_columns = {"name", "symbol", "start_date", "end_date"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Convert to dictionary format
        data = {"stocks": df.to_dict(orient="records")}

        # Write to JSON
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Portfolio JSON created at {self.json_path}")

    def json_to_csv(self) -> None:
        """Converts the JSON portfolio back to CSV (useful for editing)."""
        with open(self.json_path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data["stocks"])
        df.to_csv(self.csv_path, index=False)

        print(f"Portfolio CSV created at {self.csv_path}")

