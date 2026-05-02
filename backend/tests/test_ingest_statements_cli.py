from __future__ import annotations

import csv
import json
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import pytest

from receipts_ai import ingest_statements
from receipts_ai.ingest_statements import (
    CSV_FIELDNAMES,
    main,
    transactions_from_ofx,
    write_transaction_json,
    write_transactions_csv,
    write_transactions_json,
)
from receipts_ai.models.transaction import Kind, Source, Status, Transaction


def test_parses_sgml_ofx_bank_transactions():
    transactions = transactions_from_ofx(
        """
        OFXHEADER:100
        DATA:OFXSGML
        <OFX>
          <BANKMSGSRSV1>
            <STMTTRNRS>
              <STMTRS>
                <CURDEF>USD
                <BANKACCTFROM>
                  <BANKID>123456789
                  <ACCTID>987654321
                  <ACCTTYPE>CHECKING
                </BANKACCTFROM>
                <BANKTRANLIST>
                  <STMTTRN>
                    <TRNTYPE>DEBIT
                    <DTPOSTED>20260427000000[-8:PST]
                    <TRNAMT>-42.19
                    <FITID>2026042701
                    <NAME>COSTCO WHSE
                    <MEMO>POS PURCHASE COSTCO WHSE #123
                  </STMTTRN>
                </BANKTRANLIST>
              </STMTRS>
            </STMTTRNRS>
          </BANKMSGSRSV1>
        </OFX>
        """
    )

    assert len(transactions) == 1
    transaction = transactions[0]
    assert transaction.source == Source.bank_statement
    assert transaction.external_id == "2026042701"
    assert transaction.account_id == "123456789:987654321:CHECKING"
    assert transaction.transaction_date == date(2026, 4, 27)
    assert transaction.payee == "COSTCO WHSE"
    assert transaction.description == "POS PURCHASE COSTCO WHSE #123"
    assert transaction.amount == "-42.19"
    assert transaction.currency == "USD"
    assert transaction.kind == Kind.expense
    assert transaction.status == Status.posted
    assert transaction.receipt is None


def test_parses_xml_ofx_credit_card_transactions():
    transactions = transactions_from_ofx(
        """
        <OFX>
          <CREDITCARDMSGSRSV1>
            <CCSTMTTRNRS>
              <CCSTMTRS>
                <CURDEF>usd</CURDEF>
                <CCACCTFROM>
                  <ACCTID>4111111111111111</ACCTID>
                </CCACCTFROM>
                <BANKTRANLIST>
                  <STMTTRN>
                    <TRNTYPE>CREDIT</TRNTYPE>
                    <DTPOSTED>20260428</DTPOSTED>
                    <DTAVAIL>20260429</DTAVAIL>
                    <TRNAMT>1250.00</TRNAMT>
                    <FITID>deposit-1</FITID>
                    <NAME>Payroll</NAME>
                  </STMTTRN>
                </BANKTRANLIST>
              </CCSTMTRS>
            </CCSTMTTRNRS>
          </CREDITCARDMSGSRSV1>
        </OFX>
        """
    )

    assert len(transactions) == 1
    transaction = transactions[0]
    assert transaction.account_id == "4111111111111111"
    assert transaction.currency == "USD"
    assert transaction.posted_date == date(2026, 4, 29)
    assert transaction.amount == "1250.00"
    assert transaction.kind == Kind.income


def test_writes_transaction_csv_rows():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        external_id="fitid-1",
        account_id="acct-1",
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        description="CARD PURCHASE",
        amount="-7.00",
        currency="USD",
        kind=Kind.expense,
        status=Status.posted,
    )
    output = StringIO()

    write_transactions_csv([transaction], output)

    header, *_ = output.getvalue().splitlines()
    assert header == ",".join(CSV_FIELDNAMES)
    rows = list(csv.DictReader(StringIO(output.getvalue())))
    assert rows == [
        {
            "transaction_id": "bank_statement_1",
            "source": "bank_statement",
            "external_id": "fitid-1",
            "account_id": "acct-1",
            "transaction_date": "2026-04-27",
            "posted_date": "",
            "payee": "Coffee Shop",
            "description": "CARD PURCHASE",
            "amount": "-7.00",
            "currency": "USD",
            "kind": "expense",
            "status": "posted",
            "linked_transaction_ids": "[]",
            "category_allocations": "[]",
            "notes": "",
            "created_at": "",
            "updated_at": "",
        }
    ]


def test_transaction_json_output_uses_aliases():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        external_id="fitid-1",
        account_id="acct-1",
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
    )
    output = StringIO()

    write_transaction_json(transaction, output)

    payload = json.loads(output.getvalue())
    assert payload["externalId"] == "fitid-1"
    assert payload["accountId"] == "acct-1"
    assert payload["transactionDate"] == "2026-04-27"
    assert "receipt" not in payload


def test_transactions_json_output_writes_json_array():
    transaction_1 = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
    )
    transaction_2 = Transaction(
        id="bank_statement_2",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 28),
        payee="Payroll",
        amount="1250.00",
        currency="USD",
    )
    output = StringIO()

    write_transactions_json([transaction_1, transaction_2], output)

    payload = json.loads(output.getvalue())
    assert [transaction["id"] for transaction in payload] == [
        "bank_statement_1",
        "bank_statement_2",
    ]


def test_main_processes_multiple_statement_files_as_combined_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    statement_1_path = tmp_path / "checking.ofx"
    statement_2_path = tmp_path / "credit.ofx"
    statement_1_path.write_text(
        """
        <OFX><BANKMSGSRSV1><STMTTRNRS><STMTRS><CURDEF>USD
        <BANKACCTFROM><ACCTID>checking</ACCTID></BANKACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20260427
        <TRNAMT>-7.00<FITID>checking-1<NAME>Coffee Shop</STMTTRN></BANKTRANLIST>
        </STMTRS></STMTTRNRS></BANKMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )
    statement_2_path.write_text(
        """
        <OFX><CREDITCARDMSGSRSV1><CCSTMTTRNRS><CCSTMTRS><CURDEF>USD
        <CCACCTFROM><ACCTID>credit</ACCTID></CCACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>CREDIT<DTPOSTED>20260428
        <TRNAMT>1250.00<FITID>credit-1<NAME>Payroll</STMTTRN></BANKTRANLIST>
        </CCSTMTRS></CCSTMTTRNRS></CREDITCARDMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest_statements.py", str(statement_1_path), str(statement_2_path)],
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert [row["payee"] for row in rows] == ["Coffee Shop", "Payroll"]
    assert [row["amount"] for row in rows] == ["-7.00", "1250.00"]


def test_main_can_upsert_firestore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    statement_path = tmp_path / "checking.ofx"
    statement_path.write_text(
        """
        <OFX><BANKMSGSRSV1><STMTTRNRS><STMTRS><CURDEF>USD
        <BANKACCTFROM><ACCTID>checking</ACCTID></BANKACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20260427
        <TRNAMT>-7.00<FITID>checking-1<NAME>Coffee Shop</STMTTRN></BANKTRANLIST>
        </STMTRS></STMTTRNRS></BANKMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    def fake_upsert_transaction_to_firestore(
        transaction: Transaction, *, collection: str
    ) -> None:
        calls.append((transaction.payee, collection))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_statements.py",
            "--upsert-firestore",
            "--firestore-collection",
            "test-transactions",
            str(statement_path),
        ],
    )
    monkeypatch.setattr(
        ingest_statements,
        "upsert_transaction_to_firestore",
        fake_upsert_transaction_to_firestore,
    )

    main()

    assert calls == [("Coffee Shop", "test-transactions")]
