import pdfplumber
import re
import pandas as pd
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import json

load_dotenv()  # Load variables from .env into the environment
openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the API key
openai.api_key = openai_api_key

sections_list = ["Besondere Bestimmungen", "Kostengrundlagen", "Regiearbeiten", "Prüfungen",
                 "Baustelleneinrichtung", "Arbeitsgerüste", "Holzen und Roden", "Abbrüche und Demontagen",
                 "Sichern, unterfangen, verstärken, verschieben", "Instandsetzung und Schutz von Betonbauten",
                 "Bohren und Trennen von Beton und Mauerwerk", "Instandsetzung und Schutz von Mauerwerk aus Natursteinen",
                 "Instandhaltung und Sanierung von Abwassersystemen", "Bauarbeiten für Werkleitungen", "Rohrvortrieb",
                 "Wasserhaltung", "Baugrubenabschlüsse und Aussteifungen", "Verankerungen und Nagelwände", "Pfähle",
                 "Abdichtungen für Bauwerke unter Terrain und für Brücken", "Baugrundverbesserungen", "Garten- und Landschaftsbau",
                 "Zäune und Arealeingänge", "Sportböden für Freianlagen und Hallen", "Lärmschutzwände", "Baugruben und Erdbau",
                 "Wasserbau", "Lawinen- und Steinschlagverbau", "Altlasten, belastete Standorte und Entsorgung",
                 "Fundationsschichten für Verkehrsanlagen", "Abschlüsse, Pflästerungen, Plattendecken und Treppen", "Belagsarbeiten",
                 "Gleisbau, Sicherungsanlagen und Weichenheizungen", "Materialaufbereitung", "Zusammengefasste Leistungen im Strassen- und Leitungsbau",
                 "Kanalisationen und Entwässerungen", "Ortbetonbau", "Lager- und Fahrbahnübergänge für Brücken", "Spannsysteme",
                 "Lehr-, Schutz- und Montagegerüste", "Sprengvortrieb im Fels", "Tunnelbohrmaschinen-Vortrieb im Fels TBM",
                 "Maschinenunterstützter Vortrieb im Fels MUF", "Maschinenunterstützter Vortrieb im Lockergestein MUL",
                 "Schildmaschinen-Vortrieb im Lockergestein SM", "Ausbruchsicherungen im Untertagbau", "Bauhilfsmassnahmen im Untertagbau",
                 "Wasserhaltung im Untertagbau", "Abdichtungen im Untertagbau", "Entwässerungen im Untertagbau", "Verkleidungen im Untertagbau",
                 "Innenausbau im Untertagbau", "Kabelrohrleitungen im Untertagbau", "Vorauserkundungen und Überwachungen im UT",
                 "Fahrzeug-Rückhaltesysteme und Geländer", "Maurerarbeiten", "Vorgefertigte Elemente aus Beton und künstlichen Steinen",
                 "Verputzte Aussenwärmedämmung", "Natursteinarbeiten", "Aussenputze", "Trockenbauarbeiten: Wände", "Estriche schwimmend oder im Verbund",
                 "Bodenbeläge aus Zement, Magnesia, Kunstharz und Bitumen","Gipserarbeiten: Innenputze und Stukkaturen", "Bauheizung",
                 "Bautrocknung und Baulüftung", "Markierung auf Verkehrsflächen"]

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text(x_tolerance = 1).split("\n")
            text += "\n".join(lines)
    return text

def clean_line(line):
    line = line.strip()  # Remove leading/trailing whitespace
    line = re.sub(r'\s*\.{2,}\s*', '', line)
    line = re.sub(r'\s*-{3,}\s*', '', line)
    line = re.sub(r'\s*_{2,}\s*', '', line)
    return line
 

def is_section_header(line: str) -> bool:
    """
    Searches for npk sections in each line of text,
    when a match is found the flag is set to true.
    
    """
    for section in sections_list:
        if re.search(re.escape(section), line): #removed re.IGNORECASE
            return True  # Found a header, exit immediately
    return False  # No header found after checking all patterns and sections
 

def chunk_text(text: str) -> list[str]:
    """
    Takes text read in from pdfplumber and splits it into lines.
    Lines are cleaned with clean_line() function.
    All lines are looped through, looking for npk section headers
    defined in the the sections_list variable. When a header is found,
    a new chunk is started and all of the following lines are appended
    to this chunk until a new section header is detected. This breaks up the
    text into many (possibly short) chunks with npk-bau section themes.
    
    """
    lines = text.split('\n')
    request_chunks = []
    current_chunk = ""

    for line in lines:
        cleaned_line = clean_line(line)
        if is_section_header(cleaned_line):
            if current_chunk:
                request_chunks.append(current_chunk.strip())
                current_chunk = cleaned_line + "\n"
        elif cleaned_line:  # Only add non-empty lines to the chunk
            current_chunk += cleaned_line + "\n"
 

    if current_chunk:  # Add the last chunk
        request_chunks.append(current_chunk.strip())
 

    return request_chunks

def concatenate_by_section(text: str) -> list[str]:
    """
    
    chunk_text() is applied resulting in multiple chunks for each npk section,
    this function combines all chunks relating to each npk section resulting in
    a single (large) chunk per npk section.

    """
    
    section_map = {}
    chunks = chunk_text(text)
    current_header = None
    
    for chunk in chunks:
        cleaned_chunk = chunk.strip()
        found_header = None
        for header in sections_list:
            if re.search(re.escape(header), cleaned_chunk):
                found_header = header
                break

        if found_header:
            current_header = found_header
            if current_header not in section_map:
                section_map[current_header] = cleaned_chunk + "\n"
            else:
                section_map[current_header] += cleaned_chunk + "\n"
        elif current_header:
            section_map[current_header] = section_map.get(current_header, "") + cleaned_chunk + "\n"

    concatenated_chunks = [f"{header}\n{content.strip()}" for header, content in section_map.items()]
    return concatenated_chunks


def a_w_section(text: str) -> list[str]:
    """
    
    filters the chunks produced by concatenate_by_section()
    for only the npk sections relevant to Arthur Weber

    """
    a_w_sections = ["Bauarbeiten für Werkleitungen", "Wasserhaltung", "Kanalisationen und Entwässerungen",
                    "Ortbetonbau", "Vorgefertigte Elemente aus Beton und künstlichen Steinen",
                    "Garten- und Landschaftsbau", "Fundationsschichten für Verkehrsanlagen", "Baugruben und Erdbau"]

    a_w_chunks = []
    concatenated_chunks = concatenate_by_section(text)

    for chunk in concatenated_chunks:
        for section in a_w_sections:
            if re.match(re.escape(section), chunk):
                a_w_chunks.append(chunk)
                
    return a_w_chunks
            
def a_w_sub_section_chunks(pdf_path: str) -> list[str]:
    """
    Splits large section chunks into smaller, numbered subsections.
    Prepends the main section title (first line of each section) to each subsection
    so the model has full context when extracting items.
    """
    text = extract_text(pdf_path)
    subsections = ["100", "200", "300", "400", "500", "600", "700", "800", "900"]
    all_sub_section_chunks = []
    a_w_chunks = a_w_section(text)

    for main_chunk in a_w_chunks:
        lines = main_chunk.strip().split('\n')
        if not lines:
            continue

        section_header = lines[0].strip()  # Grab the first line (containing the npk section title)
        subsection_map = {}
        current_subsection = None

        for line in lines:
            found_subsection = None
            for s in subsections:
                if line.strip().startswith(s):
                    found_subsection = s
                    break

            if found_subsection:
                current_subsection = found_subsection
                if current_subsection not in subsection_map:
                    subsection_map[current_subsection] = section_header + "\n" + line.strip() + "\n"
                else:
                    subsection_map[current_subsection] += line.strip() + "\n"
            elif current_subsection and line.strip():
                subsection_map[current_subsection] += line.strip() + "\n"

        sub_section_chunks = [content.strip() for content in subsection_map.values()]
        all_sub_section_chunks.extend(sub_section_chunks)

    return all_sub_section_chunks

def extract_items_with_openai(chunks: list[str]) -> list[list[dict]]:
    """
    Extracts item descriptions, quantities, and units from a list of text chunks
    using the OpenAI gpt-4o-mini model.

    Args:
        chunks: A list of strings, where each string is a chunk of text
                from a building supply order document.

    Returns:
        A list of lists. Each inner list contains the extracted data for a single chunk
        in the form of a JSON array of objects (as strings that can be parsed).
        Returns an empty list for a chunk if extraction fails.
    """
    client = openai.OpenAI()
    extracted_data_all_chunks = []

    for chunk in chunks:
        input_text = chunk
        
        prompt = f"""
        You are an expert at analyzing complex construction documents in German. Your task is to **extract every ordered item** from the provided text chunk.

        ---

        ## Your Goal:
        Identify and extract **every specific item being ordered**, along with:
        - its **quantity**
        - its **unit of measurement**
        - the closest **position number**, if possible (e.g., "451.111")

        Return only the extracted data in a valid **JSON array of objects**, with the following keys:
        - `item`: full item description (string)
        - `quantity`: numeric value or "per" (string)
        - `unit`: unit of measurement (string)
        - `position`: reconstructed position number or "N/A" (string)
        - `section`: the main section header from the top of the text chunk (e.g., `"Kanalisationen und Entwässerungen"`)

        ---
        
        ## How to Identify Ordered Items:

        1. **Look for lines containing a quantity followed by a unit**, such as:
           - `- A 1000 St A`
           - `:GM 42 St`
           - `per St`, `per m`, etc.

        2. Valid units include:
           `"m"`, `"m2"`, `"m3"`, `"St"`, `"LE"`, `"kg"` and similar.

        3. **Include items with placeholders** like `"per"` (e.g., `"per m"`, `"per St"`) even if the quantity is unclear.

        4. **Do not skip items you're unsure about**. Extract them anyway and set `"quantity": "unsicher"` or `"unit": "unsicher"` as needed.

        5. **If an item is being ordered for multiple sites (e.g., `Feld A 10 St`), return the total number or quantity of items ordered.

        ---

        ## How to Build the Item Description:

        Use all relevant context above the quantity line to build a complete description:
        - Use the surrounding context above the ordered item to build the most specific item name possible.
        - Subsection headings (e.g., `.111 DN/OD 110`)
        - Item category lines (e.g., `451 Polyethylenrohre PE-R liefern und verlegen.`)
        - Main section headers (e.g., `450 Rohre und Formstücke aus Polyethylen`)
        - Other product specifications like measurements etc. (e.g. `.110 Nenn-Ringsteifigkeit SN 2, Rohrreihe S 16`)
        - If preceding lines refer to other positions (e.g., `Zu Pos. 471.111.`) use the relevant context around that position to build the description

        ---

        ## How to Find the Position (e.g., "451.111"):

        - Look at the closest preceding `.xxx` or numbered line (e.g., `.112`, `.100`)
        - Then find the closest higher-level number (e.g., `451`, `812`, etc.)
        - Combine into a `position` like `"451.112"`

        ---

        ## Do Not:
        - Do not wrap your answer in Markdown or triple backticks
        - Do not hallucinate or guess missing data; use "unsicher" if unsure

        ---

        ## Format (JSON Only, No Backticks):
        [
          {{"item": "PE Kanalrohr DN/OD 125", "quantity": "110.000", "unit": "m", "position": "451.112", "section": "Kanalisationen und Entwässerungen"}},
          {{"item": "PE Kanalbogen 160", "quantity": "2.000", "unit": "St", "position": "455.113", "section": "Kanalisationen und Entwässerungen"}},
          {{"item": "ISOPE-10 Randstellstreifen", "quantity": "per", "unit": "St", "position": "444.002", "section": "Ortbetonbau"}}
        ]

        ---

        ## Examples:

        ### Example 1
        Text:
        451 Polyethylenrohre PE-R liefern und verlegen.
        .100 PE-R mit STM, elastisch dichten.
        .110 Nenn-Ringsteifigkeit SN 2, Rohrreihe S 16.
        .111 DN/OD 110.
        - W per m A

        Result:
        {{"item": "PE Rohr DN/OD 110 SN 2, Rohrreihe S 16", "quantity": "per", "unit": "m", "position": "451.111", "section": "Kanalisationen und Entwässerungen"}}

        ---

        ### Example 2
        Text:
        444 Trennlagen und Schalldämmeinlagen.
        .002 01 Liefern und verlegen.
        04 ISOPE-10 Randstellstreifen mit
        Klettverschluss, d = 10 mm
        09 LE = m1
        - A 35.000 LE A

        Result:
        {{"item": "ISOPE-10 Randstellstreifen", "quantity": "35.000", "unit": "LE", "position": "444.002", "section": "Ortbetonbau"}}

        ---
        ### Example 3
        Text:
        611.123 01 Schachttiefe m 1.51 bis 2.00.
        99 Typ: Kontrollschacht
        Einzurechnen ist:
        - Durchlaufrinne inkl. allen
        Nebenarbeiten.
        - Gussabdeckung mit
        Geruchsverschluss Schraubbar.
        Traglast bis to 5.0.
        - Schachtleiter mit
        Einstieghalterung.
        - Sämtliche Erdarbeiten und
        aller Nebenarbeiten usw.
        - A 1.000 St A

        Result:
        {{"item": "Kontrollschacht Schachttiefe m 1.51 bis 2.00", "quantity": "1.000", "unit": "St", "position": "611.123", "section": "Ortbetonbau"}}

        ---
        ## Your Input:
        {input_text}
        """
    
        try:
            response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
            )

            llm_output = response.choices[0].message.content.strip()

            try:
                extracted_data = json.loads(llm_output)
                extracted_data_all_chunks.append(extracted_data)
            except json.JSONDecodeError:
                print(f"Error decoding output for chunk: {chunk[:50]}...\nOutput: {llm_output}")
                extracted_data_all_chunks.append([])  # Append empty list on error

        except Exception as e:
            print(f"An error occurred during OpenAI API call for chunk: {chunk[:50]}...\nError: {e}")
            extracted_data_all_chunks.append([])  # Append empty list on error

    return extracted_data_all_chunks

def main(pdf_path: str) -> pd.DataFrame:
    
    llm_chunks = a_w_sub_section_chunks(pdf_path)
    extracted_items = extract_items_with_openai(llm_chunks)

    # Flatten list of lists of dictionaries (output from extract_items_with_openai())
    flat_items = [item for sublist in extracted_items if isinstance(sublist, list) for item in sublist]

    # convert to pd df
    extracted_items_df = pd.DataFrame(flat_items)

    return extracted_items_df