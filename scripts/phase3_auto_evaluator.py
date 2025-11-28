#!/usr/bin/env python3
"""
Automated Evaluation Pipeline for Phase 3
Comprehensive quality scoring + WAF bypass validation
"""

import json
import subprocess
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

@dataclass
class ModelEvaluation:
    model_name: str
    model_path: str
    quality_score: float
    syntax_score: float
    semantic_score: float
    novelty_score: float
    waf_bypass_rate: float
    waf_scores: Dict[str, float]
    training_time: float
    gpu_memory_usage: float
    total_parameters: int
    evaluation_timestamp: str

class Phase3AutoEvaluator:
    """Automated evaluation pipeline for Phase 3 models"""
    
    def __init__(self):
        self.models_config = {
            "qwen_7b": {
                "path": "experiments/qwen_7b_4gpu_phase3",
                "test_data": "data/splits/sft_experiment/test_200_qwen.jsonl",
                "expected_params": 7000000000
            },
            "deepseek_7b": {
                "path": "experiments/deepseek_7b_4gpu_phase3", 
                "test_data": "data/splits/sft_experiment/test_200_deepseek.jsonl",
                "expected_params": 7000000000
            },
            "llama_8b": {
                "path": "experiments/llama_8b_4gpu_phase3",
                "test_data": "data/splits/sft_experiment/test_200_llama.jsonl", 
                "expected_params": 8000000000
            },
            "phi3_14b": {
                "path": "experiments/phi3_14b_4gpu_phase3",
                "test_data": "data/splits/sft_experiment/test_200_phi3.jsonl",
                "expected_params": 14000000000
            }
        }
        self.results_dir = Path("results/phase3_evaluation")
        self.results_dir.mkdir(exist_ok=True)
    
    async def evaluate_single_model(self, model_name: str) -> ModelEvaluation:
        """Evaluate a single model completely"""
        print(f"\n🚀 Starting evaluation: {model_name}")
        start_time = time.time()
        
        config = self.models_config[model_name]
        model_path = config["path"]
        test_data = config["test_data"]
        
        # Step 1: Generate model outputs
        print(f"📝 Generating outputs for {model_name}...")
        outputs_file = self.results_dir / f"{model_name}_outputs.jsonl"
        await self.generate_model_outputs(model_path, test_data, outputs_file)
        
        # Step 2: Quality evaluation
        print(f"📊 Evaluating quality for {model_name}...")
        quality_file = self.results_dir / f"{model_name}_quality.json"
        quality_scores = await self.evaluate_quality(outputs_file, quality_file)
        
        # Step 3: WAF bypass testing
        print(f"🛡️ WAF bypass testing for {model_name}...")
        waf_file = self.results_dir / f"{model_name}_waf.json"
        waf_scores = await self.evaluate_waf_bypass(outputs_file, waf_file)
        
        # Step 4: Resource usage analysis
        print(f"💾 Analyzing resource usage for {model_name}...")
        resource_stats = await self.analyze_resource_usage(model_path)
        
        evaluation_time = time.time() - start_time
        
        # Compile comprehensive evaluation
        evaluation = ModelEvaluation(
            model_name=model_name,
            model_path=model_path,
            quality_score=quality_scores["overall_quality"],
            syntax_score=quality_scores["syntax_score"], 
            semantic_score=quality_scores["semantic_score"],
            novelty_score=quality_scores["novelty_score"],
            waf_bypass_rate=waf_scores["overall_bypass_rate"],
            waf_scores=waf_scores["engine_scores"],
            training_time=resource_stats["training_time"],
            gpu_memory_usage=resource_stats["peak_memory_gb"],
            total_parameters=config["expected_params"],
            evaluation_timestamp=time.strftime("%Y%m%d_%H%M%S")
        )
        
        print(f"✅ {model_name} evaluation complete in {evaluation_time:.1f}s")
        print(f"   Quality: {evaluation.quality_score:.3f}, WAF Bypass: {evaluation.waf_bypass_rate:.3f}")
        
        return evaluation
    
    async def generate_model_outputs(self, model_path: str, test_data: str, output_file: Path):
        """Generate model outputs for test data"""
        cmd = [
            "python", "scripts/evaluate_model.py",
            "--model_path", model_path,
            "--test_data", test_data, 
            "--output_file", str(output_file),
            "--max_length", "512",
            "--batch_size", "8"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Model output generation failed: {stderr.decode()}")
    
    async def evaluate_quality(self, outputs_file: Path, quality_file: Path) -> Dict[str, float]:
        """Run quality evaluation"""
        cmd = [
            "python", "scripts/evaluate_quality.py",
            "--input_file", str(outputs_file),
            "--output_file", str(quality_file),
            "--syntax_weight", "0.6",
            "--semantic_weight", "0.3", 
            "--novelty_weight", "0.1"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode != 0:
            raise Exception("Quality evaluation failed")
        
        # Load quality results
        with open(quality_file, 'r') as f:
            quality_data = json.load(f)
        
        return {
            "overall_quality": quality_data["summary"]["overall_quality"],
            "syntax_score": quality_data["summary"]["avg_syntax_score"],
            "semantic_score": quality_data["summary"]["avg_semantic_score"], 
            "novelty_score": quality_data["summary"]["avg_novelty_score"]
        }
    
    async def evaluate_waf_bypass(self, outputs_file: Path, waf_file: Path) -> Dict[str, Any]:
        """Run WAF bypass evaluation"""
        cmd = [
            "python", "scripts/enhanced_waf_test.py",
            str(outputs_file),
            str(waf_file)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode != 0:
            raise Exception("WAF bypass evaluation failed")
        
        # Load WAF results
        with open(waf_file, 'r') as f:
            waf_data = json.load(f)
        
        return {
            "overall_bypass_rate": waf_data["metadata"]["avg_bypass_score"],
            "engine_scores": {
                "modsecurity": np.mean([r["waf_results"]["modsecurity"]["bypass_score"] for r in waf_data["results"]]),
                "cloudflare": np.mean([r["waf_results"]["cloudflare"]["bypass_score"] for r in waf_data["results"]]),
                "aws_waf": np.mean([r["waf_results"]["aws_waf"]["bypass_score"] for r in waf_data["results"]]),
                "akamai": np.mean([r["waf_results"]["akamai"]["bypass_score"] for r in waf_data["results"]])
            }
        }
    
    async def analyze_resource_usage(self, model_path: str) -> Dict[str, float]:
        """Analyze model resource usage"""
        # Extract training logs
        log_file = Path(model_path) / "trainer_state.json"
        
        try:
            with open(log_file, 'r') as f:
                trainer_state = json.load(f)
            
            training_time = trainer_state.get("log_history", [{}])[-1].get("train_runtime", 0.0)
            
            # Estimate peak memory usage (placeholder - would need actual monitoring)
            model_size_gb = 14.0  # Default estimate
            
            return {
                "training_time": training_time / 3600,  # Convert to hours
                "peak_memory_gb": model_size_gb
            }
            
        except:
            return {"training_time": 0.0, "peak_memory_gb": 14.0}
    
    async def evaluate_all_models(self) -> List[ModelEvaluation]:
        """Evaluate all Phase 3 models"""
        print("🎯 Starting Phase 3 comprehensive evaluation...")
        
        evaluations = []
        
        # Sequential evaluation to avoid resource conflicts
        for model_name in self.models_config.keys():
            try:
                evaluation = await self.evaluate_single_model(model_name)
                evaluations.append(evaluation)
                
                # Save individual result
                individual_file = self.results_dir / f"{model_name}_complete_eval.json"
                with open(individual_file, 'w') as f:
                    json.dump(asdict(evaluation), f, indent=2)
                    
            except Exception as e:
                print(f"❌ Failed to evaluate {model_name}: {e}")
                continue
        
        return evaluations
    
    def generate_comparison_report(self, evaluations: List[ModelEvaluation]):
        """Generate comprehensive comparison report"""
        
        if not evaluations:
            print("❌ No evaluations to compare")
            return
        
        # Create comparison DataFrame
        df_data = []
        for eval_result in evaluations:
            df_data.append({
                "Model": eval_result.model_name,
                "Parameters (B)": eval_result.total_parameters / 1e9,
                "Quality Score": eval_result.quality_score,
                "Syntax Score": eval_result.syntax_score,
                "Semantic Score": eval_result.semantic_score, 
                "Novelty Score": eval_result.novelty_score,
                "WAF Bypass Rate": eval_result.waf_bypass_rate,
                "ModSecurity Bypass": eval_result.waf_scores.get("modsecurity", 0),
                "Cloudflare Bypass": eval_result.waf_scores.get("cloudflare", 0),
                "AWS WAF Bypass": eval_result.waf_scores.get("aws_waf", 0),
                "Akamai Bypass": eval_result.waf_scores.get("akamai", 0),
                "Training Time (h)": eval_result.training_time,
                "Memory Usage (GB)": eval_result.gpu_memory_usage
            })
        
        df = pd.DataFrame(df_data)
        
        # Calculate composite scores
        df["Composite Score"] = (
            df["Quality Score"] * 0.4 + 
            df["WAF Bypass Rate"] * 0.35 + 
            (1 / df["Training Time (h)"]) * 0.15 + 
            (1 / df["Memory Usage (GB)"]) * 0.10
        )
        
        # Sort by composite score
        df = df.sort_values("Composite Score", ascending=False)
        
        # Generate report
        report_file = self.results_dir / "phase3_comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# 🏆 Phase 3 Model Comparison Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Models Evaluated**: {len(evaluations)}\n\n")
            
            f.write("## 📊 Overall Rankings\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n\n")
            
            f.write("## 🥇 Winner Analysis\n\n")
            winner = df.iloc[0]
            f.write(f"**🏆 Best Model: {winner['Model']}**\n")
            f.write(f"- Quality Score: {winner['Quality Score']:.3f}\n")
            f.write(f"- WAF Bypass Rate: {winner['WAF Bypass Rate']:.3f}\n") 
            f.write(f"- Training Efficiency: {winner['Training Time (h)']:.1f} hours\n")
            f.write(f"- Composite Score: {winner['Composite Score']:.3f}\n\n")
            
            f.write("## 📈 Performance Summary\n\n")
            f.write(f"- **Best Quality**: {df.loc[df['Quality Score'].idxmax(), 'Model']} ({df['Quality Score'].max():.3f})\n")
            f.write(f"- **Best WAF Bypass**: {df.loc[df['WAF Bypass Rate'].idxmax(), 'Model']} ({df['WAF Bypass Rate'].max():.3f})\n")
            f.write(f"- **Fastest Training**: {df.loc[df['Training Time (h)'].idxmin(), 'Model']} ({df['Training Time (h)'].min():.1f}h)\n")
            f.write(f"- **Most Efficient**: {df.loc[df['Memory Usage (GB)'].idxmin(), 'Model']} ({df['Memory Usage (GB)'].min():.1f}GB)\n\n")
            
            # Phase 2 vs Phase 3 comparison
            f.write("## 🚀 Phase 2 vs Phase 3 Improvement\n\n")
            best_quality = df['Quality Score'].max()
            best_bypass = df['WAF Bypass Rate'].max()
            
            f.write(f"- **Quality Improvement**: {best_quality:.3f} vs 0.463 = +{((best_quality/0.463 - 1) * 100):.1f}%\n")
            f.write(f"- **WAF Bypass Improvement**: {best_bypass:.3f} vs 0.60 = +{((best_bypass/0.60 - 1) * 100):.1f}%\n")
            
            # Success criteria
            f.write("\n## ✅ Success Criteria Met\n\n")
            targets_met = 0
            total_targets = 0
            
            criteria = [
                ("Quality Score ≥ 0.55", df['Quality Score'].max() >= 0.55),
                ("WAF Bypass Rate ≥ 0.70", df['WAF Bypass Rate'].max() >= 0.70),
                ("Training Time ≤ 10h", df['Training Time (h)'].min() <= 10),
                ("Memory Usage ≤ 20GB", df['Memory Usage (GB)'].max() <= 20)
            ]
            
            for criterion, met in criteria:
                status = "✅" if met else "❌"
                f.write(f"- {status} {criterion}\n")
                if met:
                    targets_met += 1
                total_targets += 1
            
            success_rate = targets_met / total_targets * 100
            f.write(f"\n**Overall Success Rate: {success_rate:.0f}% ({targets_met}/{total_targets})**\n")
        
        print(f"📋 Comparison report saved: {report_file}")
        print(f"🏆 Winner: {winner['Model']} (Composite Score: {winner['Composite Score']:.3f})")
        
        return df

async def main():
    """Main evaluation pipeline"""
    evaluator = Phase3AutoEvaluator()
    
    # Run comprehensive evaluation
    evaluations = await evaluator.evaluate_all_models()
    
    if evaluations:
        # Generate comparison report
        comparison_df = evaluator.generate_comparison_report(evaluations)
        
        # Save comprehensive results
        results_file = evaluator.results_dir / "phase3_complete_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(eval_result) for eval_result in evaluations], f, indent=2)
        
        print(f"\n🎯 Phase 3 evaluation complete!")
        print(f"📁 Results saved in: {evaluator.results_dir}")
        print(f"📊 {len(evaluations)} models evaluated")
    else:
        print("❌ No models successfully evaluated")

if __name__ == "__main__":
    asyncio.run(main())