"""
Inventory Agent for SAP MM Stock Verification & Automated Procurement.
Cross-checks required spare parts against SAP Materials Management (MM) stock levels,
calculates supply lead times, and generates compliant SAP MM Purchase Requisition (PR) payloads.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from src.config import SAPConfig, DEFAULT_SAP_CONFIG
from src.agents.state import InventoryItemStatus


class InventoryAgent:
    """
    Autonomous SAP MM Inventory Agent.
    Interrogates SAP S/4HANA Material Stock (`API_PRODUCT_SRV`), reserves available items,
    and automatically triggers Purchase Requisitions (`API_PURCHASEREQUISITION_PROCESS_SRV`)
    when critical maintenance components are out of stock.
    """

    DEFAULT_CATALOG: Dict[str, Dict[str, Any]] = {
        "MAT-BRG-6205-2RS": {
            "name": "Deep Groove Ball Bearing SKF 6205-2RS",
            "quantity_on_hand": 3,
            "quantity_reserved": 1,
            "lead_time_days": 5,
            "unit_cost_usd": 42.50,
            "storage_location": "SL01",
            "bin": "BIN-A-12-04",
        },
        "MAT-LUB-LGMT2": {
            "name": "SKF LGMT 2 Industrial Bearing Grease 400g Cartridge",
            "quantity_on_hand": 12,
            "quantity_reserved": 2,
            "lead_time_days": 2,
            "unit_cost_usd": 18.00,
            "storage_location": "SL01",
            "bin": "BIN-C-04-01",
        },
        "MAT-WGT-BAL-10G": {
            "name": "Rotor Dynamic Balancing Counterweight Clip Kit (10g / 20g)",
            "quantity_on_hand": 6,
            "quantity_reserved": 0,
            "lead_time_days": 3,
            "unit_cost_usd": 25.00,
            "storage_location": "SL02",
            "bin": "BIN-B-08-02",
        },
        "MAT-SHM-SS-KIT": {
            "name": "Stainless Steel Precision Slotted Machinery Shims Assortment Kit",
            "quantity_on_hand": 4,
            "quantity_reserved": 0,
            "lead_time_days": 3,
            "unit_cost_usd": 65.00,
            "storage_location": "SL02",
            "bin": "BIN-B-09-01",
        },
        "MAT-CPL-SPIDER-01": {
            "name": "Elastomeric Spider Insert for Flexible Jaw Coupling HRC-70",
            "quantity_on_hand": 5,
            "quantity_reserved": 1,
            "lead_time_days": 4,
            "unit_cost_usd": 32.00,
            "storage_location": "SL01",
            "bin": "BIN-A-14-03",
        },
        "MAT-FAN-COOL-100": {
            "name": "Electric Motor External Cooling Fan Impeller D100mm",
            "quantity_on_hand": 2,
            "quantity_reserved": 0,
            "lead_time_days": 6,
            "unit_cost_usd": 38.00,
            "storage_location": "SL01",
            "bin": "BIN-D-02-05",
        },
        "MAT-THM-PST-50G": {
            "name": "High-Conductivity Heat Sink Thermal Compound Paste (50g)",
            "quantity_on_hand": 8,
            "quantity_reserved": 1,
            "lead_time_days": 1,
            "unit_cost_usd": 15.00,
            "storage_location": "SL02",
            "bin": "BIN-C-01-02",
        },
        "MAT-BELT-V-A42": {
            "name": "Heavy Duty Industrial V-Belt Section A42",
            "quantity_on_hand": 4,
            "quantity_reserved": 0,
            "lead_time_days": 3,
            "unit_cost_usd": 22.00,
            "storage_location": "SL01",
            "bin": "BIN-A-03-01",
        },
        "MAT-SENS-MPU6050": {
            "name": "6-Axis Digital Accelerometer and Gyroscope Sensor Module MPU-6050",
            "quantity_on_hand": 2,
            "quantity_reserved": 0,
            "lead_time_days": 4,
            "unit_cost_usd": 12.50,
            "storage_location": "SL02",
            "bin": "BIN-E-01-01",
        },
    }

    def __init__(
        self,
        config: Optional[SAPConfig] = None,
        custom_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.config = config or DEFAULT_SAP_CONFIG
        self.catalog: Dict[str, Dict[str, Any]] = {}
        # Deep copy catalog to prevent cross-test state pollution
        source = custom_catalog if custom_catalog is not None else self.DEFAULT_CATALOG
        for mat_id, data in source.items():
            self.catalog[mat_id] = data.copy()

    def set_stock(self, material_id: str, quantity_on_hand: int, quantity_reserved: int = 0) -> None:
        """Helper to modify stock level for testing out-of-stock and in-stock conditions."""
        if material_id in self.catalog:
            self.catalog[material_id]["quantity_on_hand"] = quantity_on_hand
            self.catalog[material_id]["quantity_reserved"] = quantity_reserved
        else:
            self.catalog[material_id] = {
                "name": material_id,
                "quantity_on_hand": quantity_on_hand,
                "quantity_reserved": quantity_reserved,
                "lead_time_days": 7,
                "unit_cost_usd": 50.0,
                "storage_location": self.config.storage_location,
                "bin": "BIN-GEN-01",
            }

    def check_part(self, material_id: str, quantity_required: int = 1) -> InventoryItemStatus:
        """Check stock availability, reserved units, and lead time for a single material."""
        item = self.catalog.get(material_id)
        if not item:
            return InventoryItemStatus(
                material_id=material_id,
                material_name=f"Unknown Material ({material_id})",
                quantity_required=quantity_required,
                quantity_on_hand=0,
                quantity_available=0,
                is_in_stock=False,
                lead_time_days=14,
                storage_location=self.config.storage_location,
                unit_cost_usd=0.0,
            )

        on_hand = item["quantity_on_hand"]
        reserved = item["quantity_reserved"]
        available = max(0, on_hand - reserved)
        is_in_stock = available >= quantity_required

        return InventoryItemStatus(
            material_id=material_id,
            material_name=item["name"],
            quantity_required=quantity_required,
            quantity_on_hand=on_hand,
            quantity_available=available,
            is_in_stock=is_in_stock,
            lead_time_days=item["lead_time_days"] if not is_in_stock else 0,
            storage_location=item.get("storage_location", self.config.storage_location),
            unit_cost_usd=item.get("unit_cost_usd", 0.0),
        )

    def verify_inventory(
        self,
        required_parts: List[Dict[str, Any]],
        asset_id: str = "MOTOR-001",
    ) -> Dict[str, Any]:
        """
        Verify all required parts against SAP MM stock and generate Purchase Requisition if missing.

        Args:
            required_parts: List of parts required by Diagnostic Agent
            asset_id: Asset identifier

        Returns:
            Dict containing parts_available flag, inventory check items, max lead time, and SAP MM PR payload.
        """
        if not required_parts:
            return {
                "parts_available": True,
                "items": [],
                "shortages": [],
                "max_lead_time_days": 0,
                "sap_mm_pr": None,
                "summary": "No replacement components required for this maintenance action.",
            }

        item_statuses: List[InventoryItemStatus] = []
        shortages: List[InventoryItemStatus] = []
        max_lead_time_days = 0

        for part in required_parts:
            mat_id = part.get("material_id", "")
            qty_needed = int(part.get("quantity_required", 1))
            status = self.check_part(mat_id, qty_needed)
            item_statuses.append(status)

            if not status.is_in_stock:
                shortages.append(status)
                if status.lead_time_days > max_lead_time_days:
                    max_lead_time_days = status.lead_time_days

        all_available = len(shortages) == 0

        # If items are missing, generate compliant SAP MM Purchase Requisition payload
        sap_mm_pr = None
        if not all_available:
            sap_mm_pr = self.generate_purchase_requisition(shortages, asset_id)

        items_dict = [s.model_dump() for s in item_statuses]
        shortages_dict = [s.model_dump() for s in shortages]

        if all_available:
            summary = (
                f"All {len(required_parts)} required replacement parts are AVAILABLE in SAP MM "
                f"storage locations ({self.config.storage_location}). Parts reserved for Work Order."
            )
        else:
            shortage_names = ", ".join([f"{s.material_name} (Missing: {s.quantity_required - s.quantity_available})" for s in shortages])
            summary = (
                f"INVENTORY SHORTAGE DETECTED: {len(shortages)} part(s) unavailable in SAP MM: {shortage_names}. "
                f"Maximum supply lead time: {max_lead_time_days} days. "
                f"Automated SAP MM Purchase Requisition (PR) generated."
            )

        return {
            "parts_available": all_available,
            "items": items_dict,
            "shortages": shortages_dict,
            "max_lead_time_days": max_lead_time_days,
            "sap_mm_pr": sap_mm_pr,
            "summary": summary,
        }

    def generate_purchase_requisition(
        self,
        shortage_items: List[InventoryItemStatus],
        asset_id: str = "MOTOR-001",
    ) -> Dict[str, Any]:
        """
        Construct an SAP S/4HANA OData v4 Purchase Requisition payload (`API_PURCHASEREQUISITION_PROCESS_SRV`).
        """
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        pr_items = []
        for idx, item in enumerate(shortage_items, start=1):
            shortage_qty = max(1, item.quantity_required - item.quantity_available)
            delivery_date = (now + timedelta(days=item.lead_time_days)).strftime("%Y-%m-%d")

            pr_items.append({
                "PurchaseRequisitionItem": f"{idx * 10:05d}",
                "Material": item.material_id,
                "Plant": self.config.plant,
                "StorageLocation": item.storage_location,
                "RequestedQuantity": float(shortage_qty),
                "BaseUnit": "EA",
                "PurchaseRequisitionPrice": float(item.unit_cost_usd),
                "PurReqnPriceQuantity": 1.0,
                "MaterialGroup": "SPARE-PARTS",
                "PurchasingGroup": self.config.purchasing_group,
                "PurchasingOrganization": self.config.purchasing_organization,
                "DeliveryDate": delivery_date,
                "AccountAssignmentCategory": "K",  # Cost Center
                "CostCenter": self.config.cost_center,
                "PurchaseRequisitionItemText": f"Automated Predictive Replenishment for {asset_id} - {item.material_name}",
            })

        return {
            "PurchaseRequisitionType": "NB",  # Standard PR
            "PurReqnDescription": f"Auto-PR: Predictive Maintenance Spare Parts for {asset_id}",
            "CreationDate": today_str,
            "Plant": self.config.plant,
            "CostCenter": self.config.cost_center,
            "TotalItemCount": len(pr_items),
            "EstimatedTotalValueUSD": round(sum(i["PurchaseRequisitionPrice"] * i["RequestedQuantity"] for i in pr_items), 2),
            "to_PurchaseRequisitionItem": pr_items,
        }
