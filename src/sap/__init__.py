"""
SAP S/4HANA Enterprise Core Integration Module.
Provides OData v4 client execution for SAP PM Maintenance Orders and SAP MM Purchase Requisitions.
"""

from src.sap.client import SAPClient

__all__ = ["SAPClient"]
