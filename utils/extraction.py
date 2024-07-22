import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import instructor
import openai
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

PRICING = {
    "gpt-3.5-turbo": {"input": 0.5 / 1e6, "output": 1.5 / 1e6},
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.6 / 1e6},
    "gpt-4o": {"input": 5 / 1e6, "output": 15 / 1e6},
}


def calculate_pricing(usage: List[CompletionUsage], model_name: str) -> float:
    """
    Calculate pricing based on usage
    """
    total_price = 0
    for u in usage:
        total_price += u.prompt_tokens * PRICING[model_name]["input"]
        total_price += u.completion_tokens * PRICING[model_name]["output"]
    return total_price


class FileType(Enum):
    INVOICE = "Invoice"
    FAKTUR_PAJAK = "Faktur Pajak"
    PURCHASE_ORDER = "Purchase Order"
    SURAT_JALAN = "Surat Jalan"
    BERITA_ACARA = "Berita Acara"
    SPK = "Surat Penunjukan Kerja"
    OTHER = "Other"


CLASSIFICATION_PROMPT = """Classify a given document into one of the defined categories. 
NOTE THAT PRICING PROPOSAL AND PAYMENT APPROVAL FORM ARE NOT AN INVOICE!"""


class Classification(BaseModel):
    instructions: str = Field(description=CLASSIFICATION_PROMPT)
    label: FileType


class Item(BaseModel):
    name: str
    quantity: int  # = Field(default=None)
    price: float  # = Field(default=None)
    total: float  # = Field(description="Total = Price * Quantity")

    @model_validator(mode="after")
    def total_is_correct(self) -> Self:
        assert self.total == self.price * self.quantity
        return self


class Payment(BaseModel):
    vendor_name: str
    bank_name: str
    account_number: int


class Invoice(BaseModel):
    """SUBTOTAL, TAX (PPN), AND TOTAL ARE NOT ITEMS"""

    invoice_number: str
    vendor_name: str
    customer_name: str
    date_of_issue: datetime.date
    # due_date: datetime.date
    items: List[Item]
    sub_total: float
    tax: float
    total: float  # = Field(description="Total = Subtotal + Tax")
    payment: Optional[Payment]

    # @model_validator(mode="after")
    # def sub_total_is_correct(self) -> Self:
    #     assert self.sub_total == sum([item.total for item in self.items])
    #     return self


class BadanUsaha(BaseModel):
    nama: str = Field(serialization_alias="name")
    alamat: str = Field(serialization_alias="address")
    npwp: str


class FakturPajak(BaseModel):
    nomor_seri: str
    pengusaha_kena_pajak: BadanUsaha = Field(serialization_alias="vendor")
    pembeli_barang_kena_pajak: BadanUsaha = Field(serialization_alias="customer")
    items: List[Item] = Field(description="above 'Harga Jual/Penggantian' from the table")
    harga_jual: float
    dasar_pengenaan_pajak: float
    total_ppn: float
    tanggal: datetime.date
    nama_penandatangan: str
    invoice_number: Optional[str]  # = Field(description="If exists, located near 'tanggal' and 'nama penandatangan'")

    @model_validator(mode="after")
    def sub_total_is_correct(self) -> Self:
        assert self.dasar_pengenaan_pajak == sum([item.total for item in self.items])
        return self


class PurchaseOrder(BaseModel):
    po_number: str = Field(description="starts with 'PO'")
    vendor_name: str
    customer_name: str
    items: List[Item]
    total: float
    ppn: float
    grand_total: float

    @model_validator(mode="after")
    def grand_total_is_correct(self) -> Self:
        assert self.total == sum([item.total for item in self.items])
        assert self.grand_total == self.total + self.ppn
        return self


class SuratJalan(BaseModel):
    # surat_jalan_number: str
    from_name: str = Field(serialization_alias="vendor_name")
    to_name: str = Field(exclude=True)
    invoice_number: str
    # invoice_date: datetime.date
    # due_date: datetime.date


def openai_schema(response_model: BaseModel) -> Dict[str, Any]:
    return instructor.function_calls.openai_schema(response_model).openai_schema


def extract_model_output(ocr_text: str, model_output_class: BaseModel, model_name: str) -> Dict[str, Any]:
    """
    Extract information from the OCR text using the specified model and output class.
    """
    instructor_client = instructor.from_openai(client=openai.OpenAI())
    extraction_result = instructor_client.chat.completions.create(
        model=model_name,
        response_model=model_output_class,
        messages=[{"role": "user", "content": ocr_text}],
        temperature=0,
        max_retries=1,
    )
    return {
        "response": extraction_result.model_dump(by_alias=True),
        "usage": extraction_result._raw_response.usage,
    }


def classify_file_type(ocr_text: str, model_name: str) -> Dict[str, Any]:
    """
    Classify the file type using OCR text and the specified model.
    """
    classification_result = extract_model_output(ocr_text, Classification, model_name)
    classification_result["response"] = classification_result["response"]["label"].value
    return classification_result


def main_extract(ocr_text: str, selected_model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Main function to classify the file type and extract information using the specified model.
    """
    extraction_result = {}
    model_mapping = {
        "Invoice": Invoice,
        "Faktur Pajak": FakturPajak,
        "Purchase Order": PurchaseOrder,
        "Surat Jalan": SuratJalan,
    }
    usage = []

    # Classify the file type using OCR text
    file_type = classify_file_type(ocr_text, selected_model_name)
    extraction_result["file_type"] = file_type["response"]
    usage.append(file_type["usage"])

    # Get the corresponding model output class based on the file type
    model_output_class = model_mapping.get(file_type["response"], None)
    if model_output_class is not None:
        # Extract information using the model
        extracted_output = extract_model_output(ocr_text, model_output_class, selected_model_name)
        extraction_result.update(extracted_output["response"])
        usage.append(extracted_output["usage"])

    # Calculate price based on usage
    price = calculate_pricing(usage, selected_model_name)

    return {"response": extraction_result, "price": price}


# def insert_to_excel(file_path: str, new_row_df: pd.DataFrame) -> None:
#     existing_df = pd.read_excel(file_path)
#     updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
#     updated_df.to_excel(file_path, index=False)
