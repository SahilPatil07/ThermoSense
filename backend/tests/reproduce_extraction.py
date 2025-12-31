
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import asyncio

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock dependencies BEFORE importing main
sys.modules['backend.tools.storage'] = MagicMock()
sys.modules['backend.tools.sensor_harvester'] = MagicMock()
sys.modules['backend.tools.ppt_tools'] = MagicMock()
sys.modules['backend.tools.enhanced_llm_orchestrator'] = MagicMock()

# Now import the function to test
# We need to mock the global objects in main.py
with patch('backend.main.storage') as mock_storage, \
     patch('backend.main.sensor_harvester') as mock_harvester, \
     patch('backend.main.ppt_extractor') as mock_ppt_extractor:

    from backend.main import smart_extraction, ExtractionRequest

    async def test_extraction_logic():
        print("Testing Smart Extraction Logic...")
        
        # Setup Mocks
        # Create a mock for the session directory that handles path operations
        mock_session_dir = MagicMock()
        mock_storage.get_session_dir.return_value = mock_session_dir
        
        # Handle / operator (truediv) and joinpath
        def create_mock_path(name):
            p = MagicMock()
            p.exists.return_value = True
            p.lower.return_value = str(name).lower()
            p.name = str(name)
            p.__str__.return_value = f"c:/Thermosense/workspace/test/{name}"
            return p

        mock_session_dir.__truediv__.side_effect = create_mock_path
        mock_session_dir.joinpath.side_effect = create_mock_path
        
        # Mock Sensor Harvester to return some data
        mock_df = pd.DataFrame({'Time': [1, 2], 'Temp': [25, 26]})
        mock_harvester.harvest_sensors.return_value = (mock_df, {'Temp': 'file1.xlsx'})
        
        # Mock PPT Extractor to return some slides
        mock_ppt_extractor.extract_slides.return_value = [{
            'slide_number': 1, 
            'title': 'Slide 1', 
            'image_path': 'slide1.png',
            'extraction_mode': 'full'
        }]
        
        # Case 1: No Section Provided
        print("\n--- Case 1: No Section Provided ---")
        req = ExtractionRequest(
            session_id="test",
            filenames=["data.xlsx", "slides.pptx"],
            sensors=["Temp"],
            section=None
        )

        # Run the function
        await smart_extraction(req)
        
        # Verify PPT extraction
        # Should call add_to_report_section with "Appendices" (default)
        ppt_calls = [call for call in mock_storage.add_to_report_section.call_args_list 
                     if call.kwargs.get('item_type') == 'image']
        
        if ppt_calls:
            print("✅ PPT Extraction called add_to_report_section")
            section_used = ppt_calls[0].kwargs.get('section') or ppt_calls[0].args[1]
            print(f"   Section used: {section_used}")
            if section_used == "Appendices":
                print("   ✅ Correctly defaulted to 'Appendices'")
            else:
                print(f"   ❌ Unexpected default: {section_used}")
        else:
            print("❌ PPT Extraction DID NOT call add_to_report_section")

        # Verify Excel Extraction
        # Should NOT call add_to_report_section because section is None (Current Bug/Feature)
        excel_calls = [call for call in mock_storage.add_to_report_section.call_args_list 
                       if call.kwargs.get('item_type') in ['text', 'table']]
        
        if not excel_calls:
            print("✅ Excel Extraction correctly SKIPPED report addition (Current Behavior)")
        else:
            print("❌ Excel Extraction ADDED to report (Unexpected for current code)")
            for call in excel_calls:
                print(f"   Call: {call}")

    if __name__ == "__main__":
        asyncio.run(test_extraction_logic())
