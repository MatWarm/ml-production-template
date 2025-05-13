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


class Abbreviation(Base):
    __tablename__ = "abbreviation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ABBREVIATION_CODE = Column(String(50))
    ABBREVIATED = Column(String(50))
    STANDARD_DESCRIPTION = Column(String(50))


class Grammar(Base):
    __tablename__ = "grammar"
    id = Column(Integer, primary_key=True, autoincrement=True)
    WORD = Column(String(50))
    DESCRIPTION = Column(String(50))
    COMPOUND = Column(String(50))
    LEMMA = Column(String(50))
    STEM = Column(String(50))
    PART_OF_SPEECH = Column(String(50))


class Morphosyntactic(Base):
    __tablename__ = "morphosyntactic"
    id = Column(Integer, primary_key=True, autoincrement=True)
    WORD = Column(String(50))
    DESCRIPTION = Column(String(50))
    COMPOUND = Column(String(50))
    LEMMA = Column(String(50))
    STEM = Column(String(50))
    PART_OF_SPEECH = Column(String(50))


class TermBanks(Base):
    __tablename__ = "term_banks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    WORD = Column(String(50))
    EXAMPLE = Column(String(50))


def load_data_aircraft_annotation(item):
    annotation = Annotation(
        ANNOTATION_NUMBER=item["IDENT"], ACTION=item["ACTION"], PROBLEM=item["PROBLEM"]
    )
    return annotation


def load_data_abbreviation(item):
    annotation = Abbreviation(
        Abbreviation_Code=item["ABBREVIATION_CODE"],
        Abbreviated=item["ABBREVIATED"],
        Standard_Description=item["STANDARD_DESCRIPTION"],
    )
    return annotation


def load_data_grammar(item):
    annotation = Grammar(
        word=item["WORD"],
        Description=item["DESCRIPTION"],
        Compound=item["COMPOUND"],
        Lemma=item["LEMMA"],
        Stem=item["STEM"],
        Part_of_Speech=item["PART_OF_SPEECH"],
    )

    return annotation


def load_data_morphosyntactic(item):
    annotation = Morphosyntactic(
        word=item["WORD"],
        Description=item["DESCRIPTION"],
        Compound=item["COMPOUND"],
        Lemma=item["LEMMA"],
        Stem=item["STEM"],
        Part_of_Speech=item["PART_OF_SPEECH"],
    )
    return annotation


def load_data_term_banks(item):
    annotation = TermBanks(word=item["WORD"], Example=item["EXAMPLE"])
    return annotation


engine = create_engine("sqlite:///data/data.db")
Session = sessionmaker(bind=engine)

session = Session()


def store_data():
    for file in Path("data/raw").glob("*.csv"):
        print(f"Processing {file.name}...")
        df = pd.read_csv(f"data/raw/{file.name}")
        with Session() as session:
            for _, row in df.iterrows():

                match file.name:
                    case "Aircraft_Annotation_DataFile.csv":
                        data = load_data_aircraft_annotation(row)
                    case "Abbreviation.csv":
                        data = load_data_abbreviation(row)
                    case "Grammar.csv":
                        data = load_data_grammar(row)
                    case "Morphosyntactic.csv":
                        data = load_data_morphosyntactic(row)
                    case "Term_Banks.csv":
                        data = load_data_term_banks(row)

                session.add(data)
                session.commit()
        print(f"Data from {file.name} stored successfully.")
        print("===")


if __name__ == "__main__":
    store_data()
    inspector = inspect(engine)
    print("Tables réellement créées :", inspector.get_table_names())
