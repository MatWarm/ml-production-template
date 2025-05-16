from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import inspect
import pandas as pd

# --- SQL setup ---
Base = declarative_base()


class Annotation(Base):
    __tablename__ = "annotation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ANNOTATION_NUMBER = Column(Integer)
    ACTION = Column(Text)
    PROBLEM = Column(Text)


def load_data_aircraft_annotation(item):
    annotation = Annotation(
        ANNOTATION_NUMBER=item["IDENT"], 
        ACTION=item["ACTION"], 
        PROBLEM=item["PROBLEM"]
    )
    return annotation


engine = create_engine("sqlite:///data/raw/data.db")
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

session = Session()


def store_data():
    for file in Path("data/raw").glob("*.csv"):
        print(f"Processing {file.name}...")
        df = pd.read_csv(f"data/raw/{file.name}")
        i = 0
        with Session() as session:
            for page_number, row in df.iterrows():
                i += 1
                match file.name:
                    case "Aircraft_Annotation_DataFile.csv":
                        data = load_data_aircraft_annotation(row) 
                    
                session.add(data)
                session.commit()
        print(i)
        print(f"Data from {file.name} stored successfully.")
        print("===")


if __name__ == "__main__":
    store_data()
    inspector = inspect(engine)
    print("Tables réellement créées :", inspector.get_table_names())
