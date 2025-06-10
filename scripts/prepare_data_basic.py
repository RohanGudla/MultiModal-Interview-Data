#!/usr/bin/env python3
"""
ITERATION 1: Basic data pipeline implementation and analysis.
Focus: Get data loading working and understand what we have.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config

class BasicDataAnalyzer:
    """Basic data pipeline for initial analysis."""
    
    def __init__(self):
        self.config = Config()
        self.results = {
            "videos": {},
            "annotations": {},
            "data_quality": {},
            "errors": [],
            "statistics": {}
        }
        
    def analyze_videos(self) -> Dict:
        """Analyze video files and extract basic information."""
        print("🎬 ANALYZING VIDEO FILES")
        print("=" * 50)
        
        video_dir = self.config.get_data_paths()["videos"]
        video_files = list(video_dir.glob("*.mp4"))
        
        print(f"Found {len(video_files)} video files:")
        
        for video_path in video_files:
            print(f"\n📹 Processing: {video_path.name}")
            
            try:
                # Extract participant ID
                participant_id = self._extract_participant_id(video_path.name)
                
                # Get basic file info
                file_size_mb = video_path.stat().st_size / (1024*1024)
                file_exists = video_path.exists()
                
                # Store basic video info (detailed analysis will come later)
                video_info = {
                    "participant_id": participant_id,
                    "file_path": str(video_path),
                    "file_size_mb": file_size_mb,
                    "file_exists": file_exists,
                    "file_extension": video_path.suffix
                }
                
                self.results["videos"][participant_id] = video_info
                
                print(f"   ✅ Participant: {participant_id}")
                print(f"   💾 File size: {file_size_mb:.1f} MB")
                print(f"   📁 Exists: {file_exists}")
                
            except Exception as e:
                error_msg = f"Error processing {video_path}: {str(e)}"
                print(f"❌ {error_msg}")
                self.results["errors"].append(error_msg)
                
        return self.results["videos"]
    
    def analyze_annotations(self) -> Dict:
        """Analyze annotation files and extract emotion data."""
        print("\n📊 ANALYZING ANNOTATION DATA")
        print("=" * 50)
        
        annotation_path = self.config.get_data_paths()["annotations"]
        
        try:
            # Check if annotation file exists
            if not annotation_path.exists():
                error_msg = f"Annotation file not found: {annotation_path}"
                print(f"❌ {error_msg}")
                self.results["errors"].append(error_msg)
                return {}
            
            print(f"Loading annotations from: {annotation_path}")
            
            # Try to read the CSV file
            try:
                # First, let's examine the file structure
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline().strip() for _ in range(15)]
                
                print(f"📄 File preview (first 15 lines):")
                for i, line in enumerate(first_lines):
                    print(f"   {i+1:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")
                
                # Find where actual data starts (skip metadata)
                data_start_row = 0
                for i, line in enumerate(first_lines):
                    if line.startswith('#DATA') or line.startswith('Study Name'):
                        data_start_row = i + 1
                        break
                
                print(f"📍 Data appears to start at row: {data_start_row}")
                
                # Load the data
                df = pd.read_csv(annotation_path, skiprows=data_start_row)
                
                print(f"✅ Loaded annotations: {len(df)} rows, {len(df.columns)} columns")
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                print(f"📋 Column names sample:")
                for i, col in enumerate(df.columns[:10]):
                    print(f"   {i+1:2d}: {col}")
                if len(df.columns) > 10:
                    print(f"   ... and {len(df.columns) - 10} more columns")
                
                # Check for participant data
                if 'Respondent Name' in df.columns:
                    participants_in_annotations = df['Respondent Name'].unique()
                    print(f"🎭 Participants in annotations: {list(participants_in_annotations)}")
                else:
                    print(f"⚠️  'Respondent Name' column not found")
                    print(f"Available columns: {list(df.columns)}")
                
                # Look for emotion-related columns
                emotion_keywords = ['Anger', 'Joy', 'Fear', 'Sadness', 'Surprise', 'Disgust', 'Contempt', 'Attention']
                emotion_columns = []
                
                for col in df.columns:
                    if any(keyword in col for keyword in emotion_keywords):
                        emotion_columns.append(col)
                
                print(f"🎭 Found {len(emotion_columns)} emotion-related columns:")
                for col in emotion_columns[:10]:  # Show first 10
                    print(f"   - {col}")
                if len(emotion_columns) > 10:
                    print(f"   ... and {len(emotion_columns) - 10} more")
                
                # Extract data for participants with videos
                video_participants = list(self.results["videos"].keys())
                print(f"🔗 Video participants: {video_participants}")
                
                for participant_id in video_participants:
                    if 'Respondent Name' in df.columns:
                        participant_data = df[df['Respondent Name'] == participant_id]
                        
                        if participant_data.empty:
                            error_msg = f"No annotation data for participant: {participant_id}"
                            print(f"⚠️  {error_msg}")
                            self.results["errors"].append(error_msg)
                            continue
                        
                        # Get the first record (should be only one)
                        record = participant_data.iloc[0]
                        
                        # Extract available data
                        annotation_info = {
                            "participant_id": participant_id,
                            "raw_data_available": True,
                            "emotion_columns_count": len(emotion_columns),
                            "sample_values": {}
                        }
                        
                        # Sample some emotion values
                        for col in emotion_columns[:5]:  # First 5 emotion columns
                            if pd.notna(record[col]):
                                annotation_info["sample_values"][col] = float(record[col])
                            else:
                                annotation_info["sample_values"][col] = None
                        
                        self.results["annotations"][participant_id] = annotation_info
                        
                        print(f"\n👤 {participant_id}:")
                        print(f"   ✅ Annotation data found")
                        print(f"   📊 {len(emotion_columns)} emotion metrics available")
                        if annotation_info["sample_values"]:
                            print(f"   📈 Sample values:")
                            for col, val in list(annotation_info["sample_values"].items())[:3]:
                                print(f"      {col}: {val}")
                
            except UnicodeDecodeError:
                # Try different encoding
                df = pd.read_csv(annotation_path, skiprows=10, encoding='latin-1')
                print(f"✅ Loaded with latin-1 encoding: {len(df)} rows")
                
        except Exception as e:
            error_msg = f"Error processing annotations: {str(e)}"
            print(f"❌ {error_msg}")
            self.results["errors"].append(error_msg)
            import traceback
            traceback.print_exc()
            
        return self.results["annotations"]
    
    def create_data_quality_report(self) -> Dict:
        """Create comprehensive data quality analysis."""
        print("\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)
        
        quality_report = {
            "video_quality": {},
            "annotation_quality": {},
            "alignment_check": {},
            "recommendations": []
        }
        
        # Video quality analysis
        if self.results["videos"]:
            video_data = list(self.results["videos"].values())
            
            quality_report["video_quality"] = {
                "total_videos": len(video_data),
                "all_files_exist": all(v["file_exists"] for v in video_data),
                "avg_file_size_mb": np.mean([v["file_size_mb"] for v in video_data]),
                "total_size_mb": sum(v["file_size_mb"] for v in video_data)
            }
            
            print(f"📹 Video Quality Summary:")
            print(f"   Total videos: {quality_report['video_quality']['total_videos']}")
            print(f"   All files exist: {quality_report['video_quality']['all_files_exist']}")
            print(f"   Avg file size: {quality_report['video_quality']['avg_file_size_mb']:.1f} MB")
            print(f"   Total size: {quality_report['video_quality']['total_size_mb']:.1f} MB")
        
        # Annotation quality analysis
        if self.results["annotations"]:
            annotation_data = list(self.results["annotations"].values())
            
            quality_report["annotation_quality"] = {
                "total_participants": len(annotation_data),
                "all_have_data": all(a["raw_data_available"] for a in annotation_data),
                "avg_emotion_columns": np.mean([a["emotion_columns_count"] for a in annotation_data]) if annotation_data else 0
            }
            
            print(f"📊 Annotation Quality Summary:")
            print(f"   Participants with annotations: {quality_report['annotation_quality']['total_participants']}")
            print(f"   All have data: {quality_report['annotation_quality']['all_have_data']}")
            print(f"   Avg emotion columns: {quality_report['annotation_quality']['avg_emotion_columns']:.0f}")
        
        # Alignment check
        video_participants = set(self.results["videos"].keys())
        annotation_participants = set(self.results["annotations"].keys())
        
        quality_report["alignment_check"] = {
            "both_data": list(video_participants & annotation_participants),
            "video_only": list(video_participants - annotation_participants),
            "annotation_only": list(annotation_participants - video_participants),
            "alignment_rate": len(video_participants & annotation_participants) / len(video_participants | annotation_participants) if video_participants | annotation_participants else 0
        }
        
        print(f"🔗 Data Alignment:")
        print(f"   Both video + annotations: {len(quality_report['alignment_check']['both_data'])}")
        print(f"   Video only: {len(quality_report['alignment_check']['video_only'])}")
        print(f"   Annotation only: {len(quality_report['alignment_check']['annotation_only'])}")
        print(f"   Alignment rate: {quality_report['alignment_check']['alignment_rate']:.1%}")
        
        # Generate recommendations
        recommendations = []
        
        if quality_report["alignment_check"]["alignment_rate"] < 1.0:
            recommendations.append("Some participants missing either video or annotation data")
        
        if not quality_report["video_quality"].get("all_files_exist", True):
            recommendations.append("Some video files are missing or inaccessible")
        
        if len(self.results["errors"]) > 0:
            recommendations.append(f"Address {len(self.results['errors'])} errors before proceeding")
        
        if len(quality_report["alignment_check"]["both_data"]) < 3:
            recommendations.append("Less than 3 participants with complete data - may need more data")
        
        quality_report["recommendations"] = recommendations
        
        if recommendations:
            print(f"⚠️  Recommendations:")
            for rec in recommendations:
                print(f"   - {rec}")
        else:
            print(f"✅ No major data quality issues detected")
        
        self.results["data_quality"] = quality_report
        return quality_report
    
    def save_results(self, save_dir: Path):
        """Save all analysis results."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results_file = save_dir / "iteration1_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Saved detailed results to: {results_file}")
        
        # Create summary report
        summary = {
            "iteration": 1,
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary": {
                "total_videos": len(self.results["videos"]),
                "total_annotations": len(self.results["annotations"]),
                "aligned_participants": len(set(self.results["videos"].keys()) & set(self.results["annotations"].keys())),
                "total_errors": len(self.results["errors"]),
                "data_quality_score": self._calculate_quality_score()
            },
            "next_steps": self._generate_next_steps()
        }
        
        summary_file = save_dir / "iteration1_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Saved summary to: {summary_file}")
        
        return summary
    
    def _extract_participant_id(self, filename: str) -> str:
        """Extract participant ID from filename."""
        # Handle patterns like "Screen recording 1 - CP 0636.mp4"
        parts = filename.replace('.mp4', '').split(' - ')
        if len(parts) > 1:
            return parts[-1].strip()
        return filename.replace('.mp4', '')
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 0.0
        max_score = 0.0
        
        # Video availability (25%)
        max_score += 0.25
        if self.results["videos"]:
            score += 0.25
        
        # Annotation availability (25%)
        max_score += 0.25
        if self.results["annotations"]:
            score += 0.25
        
        # Data alignment (25%)
        max_score += 0.25
        if self.results.get("data_quality", {}).get("alignment_check", {}).get("alignment_rate", 0) > 0:
            score += 0.25 * self.results["data_quality"]["alignment_check"]["alignment_rate"]
        
        # Error rate (25%)
        max_score += 0.25
        if len(self.results["errors"]) == 0:
            score += 0.25
        elif len(self.results["errors"]) < 3:
            score += 0.125
        
        return score / max_score if max_score > 0 else 0.0
    
    def _generate_next_steps(self) -> List[str]:
        """Generate recommended next steps based on results."""
        next_steps = []
        
        aligned_count = len(set(self.results["videos"].keys()) & set(self.results["annotations"].keys()))
        
        if aligned_count >= 3:
            next_steps.append("✅ Sufficient data for training - proceed to Iteration 2 (CNN baseline)")
        else:
            next_steps.append("⚠️ Limited aligned data - consider data augmentation or relaxed requirements")
        
        if len(self.results["errors"]) > 0:
            next_steps.append(f"🔧 Fix {len(self.results['errors'])} errors before proceeding")
        
        if len(self.results["videos"]) == 0:
            next_steps.append("🎬 No video files found - check video directory path")
        
        if len(self.results["annotations"]) == 0:
            next_steps.append("📊 No annotation data found - check annotation file path and format")
        
        return next_steps

def main():
    """Run Iteration 1: Basic data pipeline and analysis."""
    print("🚀 ITERATION 1: FOUNDATION & DATA PIPELINE")
    print("=" * 80)
    print("Goal: Understand our data and get basic pipeline working")
    print("=" * 80)
    
    # Create analyzer
    analyzer = BasicDataAnalyzer()
    
    # Create output directory
    output_dir = Path("experiments/iteration1_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Analyze videos
        video_results = analyzer.analyze_videos()
        
        # Step 2: Analyze annotations
        annotation_results = analyzer.analyze_annotations()
        
        # Step 3: Data quality report
        quality_report = analyzer.create_data_quality_report()
        
        # Step 4: Save results and generate summary
        summary = analyzer.save_results(output_dir)
        
        # Print final summary
        print(f"\n🎯 ITERATION 1 SUMMARY")
        print("=" * 50)
        print(f"✅ Videos analyzed: {summary['summary']['total_videos']}")
        print(f"✅ Annotations processed: {summary['summary']['total_annotations']}")
        print(f"✅ Aligned participants: {summary['summary']['aligned_participants']}")
        print(f"⚠️  Errors encountered: {summary['summary']['total_errors']}")
        print(f"📊 Data quality score: {summary['summary']['data_quality_score']:.2f}/1.00")
        
        if len(analyzer.results["errors"]) > 0:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for i, error in enumerate(analyzer.results["errors"], 1):
                print(f"   {i}. {error}")
        
        print(f"\n📋 NEXT STEPS:")
        for step in summary['next_steps']:
            print(f"   {step}")
        
        print(f"\n📁 All results saved to: {output_dir}")
        
        # Determine if ready for next iteration
        if summary['summary']['data_quality_score'] >= 0.5 and summary['summary']['aligned_participants'] >= 1:
            print(f"\n🎉 ITERATION 1 SUCCESS!")
            print(f"✅ Ready to proceed to Iteration 2: CNN Baseline")
            return True
        else:
            print(f"\n⚠️  ITERATION 1 NEEDS ATTENTION")
            print(f"❌ Address issues before proceeding to Iteration 2")
            return False
        
    except Exception as e:
        print(f"\n💥 ITERATION 1 FAILED")
        print(f"❌ Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)