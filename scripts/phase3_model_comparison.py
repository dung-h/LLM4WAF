#!/usr/bin/env python3
"""
Comprehensive Model Comparison Framework for Phase 3
Advanced scoring system with multiple evaluation dimensions
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

@dataclass
class ComparisonWeights:
    """Configurable weights for model comparison"""
    quality: float = 0.40      # Quality score importance
    waf_bypass: float = 0.35   # WAF bypass capability  
    efficiency: float = 0.15   # Training efficiency
    resource: float = 0.10     # Resource utilization

@dataclass
class ModelRanking:
    """Model ranking with detailed breakdown"""
    rank: int
    model_name: str
    composite_score: float
    quality_rank: int
    waf_rank: int  
    efficiency_rank: int
    resource_rank: int
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str

class Phase3ModelComparison:
    """Advanced model comparison framework"""
    
    def __init__(self, weights: ComparisonWeights = None):
        self.weights = weights or ComparisonWeights()
        self.results_dir = Path("results/phase3_evaluation")
        self.comparison_dir = Path("results/phase3_comparison") 
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.thresholds = {
            "quality_excellent": 0.70,
            "quality_good": 0.55,
            "quality_fair": 0.40,
            "waf_excellent": 0.80,
            "waf_good": 0.70, 
            "waf_fair": 0.60,
            "training_fast": 6.0,      # hours
            "training_medium": 10.0,
            "memory_efficient": 16.0,   # GB
            "memory_moderate": 24.0
        }
    
    def load_evaluations(self) -> List[Dict[str, Any]]:
        """Load all model evaluation results"""
        evaluations = []
        
        # Look for individual evaluation files
        for eval_file in self.results_dir.glob("*_complete_eval.json"):
            try:
                with open(eval_file, 'r') as f:
                    evaluation = json.load(f)
                    evaluations.append(evaluation)
            except Exception as e:
                print(f"⚠️  Failed to load {eval_file}: {e}")
        
        # Fallback: load complete results file
        if not evaluations:
            complete_file = self.results_dir / "phase3_complete_results.json"
            if complete_file.exists():
                with open(complete_file, 'r') as f:
                    evaluations = json.load(f)
        
        return evaluations
    
    def calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite scores with multiple weighting strategies"""
        
        # Normalize scores (0-1 scale)
        df_norm = df.copy()
        
        # Quality metrics (higher = better)
        df_norm["quality_norm"] = df["quality_score"] / df["quality_score"].max()
        df_norm["waf_norm"] = df["waf_bypass_rate"] / df["waf_bypass_rate"].max()
        
        # Efficiency metrics (lower = better, so invert)
        df_norm["efficiency_norm"] = (df["training_time"].max() - df["training_time"]) / (df["training_time"].max() - df["training_time"].min())
        df_norm["resource_norm"] = (df["gpu_memory_usage"].max() - df["gpu_memory_usage"]) / (df["gpu_memory_usage"].max() - df["gpu_memory_usage"].min())
        
        # Handle edge cases (single model)
        df_norm = df_norm.fillna(0.5)
        
        # Calculate weighted composite score
        df["composite_score"] = (
            df_norm["quality_norm"] * self.weights.quality +
            df_norm["waf_norm"] * self.weights.waf_bypass +
            df_norm["efficiency_norm"] * self.weights.efficiency +
            df_norm["resource_norm"] * self.weights.resource
        )
        
        # Alternative scoring strategies
        df["quality_focused"] = (
            df_norm["quality_norm"] * 0.60 +
            df_norm["waf_norm"] * 0.30 + 
            df_norm["efficiency_norm"] * 0.05 +
            df_norm["resource_norm"] * 0.05
        )
        
        df["performance_focused"] = (
            df_norm["quality_norm"] * 0.25 +
            df_norm["waf_norm"] * 0.45 +
            df_norm["efficiency_norm"] * 0.20 +
            df_norm["resource_norm"] * 0.10
        )
        
        df["efficiency_focused"] = (
            df_norm["quality_norm"] * 0.30 +
            df_norm["waf_norm"] * 0.25 +
            df_norm["efficiency_norm"] * 0.35 +
            df_norm["resource_norm"] * 0.10
        )
        
        return df
    
    def analyze_model_strengths(self, model_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze model strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Quality analysis
        quality = model_data["quality_score"]
        if quality >= self.thresholds["quality_excellent"]:
            strengths.append("Excellent payload quality")
        elif quality >= self.thresholds["quality_good"]:
            strengths.append("Good payload quality")
        elif quality < self.thresholds["quality_fair"]:
            weaknesses.append("Below-average payload quality")
        
        # WAF bypass analysis  
        waf_bypass = model_data["waf_bypass_rate"]
        if waf_bypass >= self.thresholds["waf_excellent"]:
            strengths.append("Superior WAF evasion")
        elif waf_bypass >= self.thresholds["waf_good"]:
            strengths.append("Strong WAF bypass capability")
        elif waf_bypass < self.thresholds["waf_fair"]:
            weaknesses.append("Limited WAF evasion ability")
        
        # Training efficiency
        training_time = model_data["training_time"]
        if training_time <= self.thresholds["training_fast"]:
            strengths.append("Fast training convergence")
        elif training_time > self.thresholds["training_medium"]:
            weaknesses.append("Slow training speed")
        
        # Resource efficiency
        memory = model_data["gpu_memory_usage"]
        if memory <= self.thresholds["memory_efficient"]:
            strengths.append("Memory efficient")
        elif memory > self.thresholds["memory_moderate"]:
            weaknesses.append("High memory usage")
        
        # WAF engine specific analysis
        waf_scores = model_data.get("waf_scores", {})
        best_engine = max(waf_scores.keys(), key=lambda k: waf_scores[k]) if waf_scores else None
        worst_engine = min(waf_scores.keys(), key=lambda k: waf_scores[k]) if waf_scores else None
        
        if best_engine and waf_scores[best_engine] > 0.80:
            strengths.append(f"Excellent {best_engine} bypass")
        if worst_engine and waf_scores[worst_engine] < 0.50:
            weaknesses.append(f"Poor {worst_engine} evasion")
        
        return strengths, weaknesses
    
    def generate_recommendations(self, model_data: Dict[str, Any], rank: int) -> str:
        """Generate actionable recommendations for each model"""
        
        quality = model_data["quality_score"]
        waf_bypass = model_data["waf_bypass_rate"]
        training_time = model_data["training_time"]
        
        if rank == 1:
            return "🏆 Primary production candidate. Deploy immediately with full monitoring."
        elif rank == 2:
            return "🥈 Strong backup candidate. Consider for A/B testing or specialized scenarios."
        elif quality >= 0.60 and waf_bypass >= 0.70:
            return "💎 High potential. Investigate training optimizations or ensemble approaches."
        elif quality >= 0.50 and waf_bypass >= 0.65:
            return "⚡ Moderate performer. Suitable for specific attack scenarios or further fine-tuning."
        elif training_time <= 6.0:
            return "🚀 Fast trainer. Good for rapid iteration and experimentation."
        else:
            return "🔧 Development only. Use for ablation studies or architecture experiments."
    
    def create_detailed_rankings(self, df: pd.DataFrame) -> List[ModelRanking]:
        """Create detailed model rankings with analysis"""
        
        rankings = []
        
        # Sort by composite score
        df_sorted = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        
        for idx, row in df_sorted.iterrows():
            # Calculate individual dimension ranks
            quality_rank = df.sort_values("quality_score", ascending=False).index.get_loc(row.name) + 1
            waf_rank = df.sort_values("waf_bypass_rate", ascending=False).index.get_loc(row.name) + 1
            eff_rank = df.sort_values("training_time", ascending=True).index.get_loc(row.name) + 1
            res_rank = df.sort_values("gpu_memory_usage", ascending=True).index.get_loc(row.name) + 1
            
            # Analyze strengths and weaknesses
            model_dict = row.to_dict()
            strengths, weaknesses = self.analyze_model_strengths(model_dict)
            recommendation = self.generate_recommendations(model_dict, idx + 1)
            
            ranking = ModelRanking(
                rank=idx + 1,
                model_name=row["model_name"],
                composite_score=row["composite_score"],
                quality_rank=quality_rank,
                waf_rank=waf_rank,
                efficiency_rank=eff_rank,
                resource_rank=res_rank,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendation=recommendation
            )
            
            rankings.append(ranking)
        
        return rankings
    
    def generate_comparison_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate pairwise comparison matrix"""
        
        models = df["model_name"].tolist()
        n_models = len(models)
        
        # Initialize comparison matrix
        comparison_matrix = pd.DataFrame(
            index=models,
            columns=models,
            data=0.0
        )
        
        for i, model_i in enumerate(models):
            for j, model_j in enumerate(models):
                if i != j:
                    # Calculate win probability based on composite scores
                    score_i = df[df["model_name"] == model_i]["composite_score"].iloc[0]
                    score_j = df[df["model_name"] == model_j]["composite_score"].iloc[0]
                    
                    # Softmax-style probability
                    win_prob = 1 / (1 + np.exp(-(score_i - score_j) * 10))
                    comparison_matrix.loc[model_i, model_j] = win_prob
                else:
                    comparison_matrix.loc[model_i, model_j] = 0.5
        
        return comparison_matrix
    
    def create_visualizations(self, df: pd.DataFrame, rankings: List[ModelRanking]):
        """Create comprehensive visualizations"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Phase 3 Model Comparison Analysis", fontsize=16, fontweight='bold')
        
        # 1. Composite Score Comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(df["model_name"], df["composite_score"], 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title("Overall Composite Scores")
        ax1.set_ylabel("Composite Score")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Quality vs WAF Performance
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df["quality_score"], df["waf_bypass_rate"], 
                             s=200, c=df["composite_score"], 
                             cmap='viridis', alpha=0.7)
        
        for i, model in enumerate(df["model_name"]):
            ax2.annotate(model, (df["quality_score"].iloc[i], df["waf_bypass_rate"].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel("Quality Score")
        ax2.set_ylabel("WAF Bypass Rate")
        ax2.set_title("Quality vs WAF Performance")
        ax2.axhline(y=0.70, color='r', linestyle='--', alpha=0.5, label='WAF Target')
        ax2.axvline(x=0.55, color='r', linestyle='--', alpha=0.5, label='Quality Target')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='Composite Score')
        
        # 3. Training Efficiency
        ax3 = axes[0, 2]
        ax3.barh(df["model_name"], df["training_time"], color='lightcoral')
        ax3.set_xlabel("Training Time (hours)")
        ax3.set_title("Training Efficiency")
        ax3.axvline(x=10, color='r', linestyle='--', alpha=0.5, label='Target: <10h')
        ax3.legend()
        
        # 4. WAF Engine Breakdown
        ax4 = axes[1, 0]
        waf_engines = ["modsecurity", "cloudflare", "aws_waf", "akamai"]
        waf_data = []
        
        for _, row in df.iterrows():
            waf_scores = row.get("waf_scores", {})
            waf_data.append([waf_scores.get(engine, 0) for engine in waf_engines])
        
        waf_df = pd.DataFrame(waf_data, columns=waf_engines, index=df["model_name"])
        waf_df.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title("WAF Engine Bypass Performance")
        ax4.set_ylabel("Bypass Rate")
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title="WAF Engine")
        
        # 5. Resource Usage
        ax5 = axes[1, 1]
        ax5.scatter(df["training_time"], df["gpu_memory_usage"], 
                   s=df["total_parameters"]/1e8, alpha=0.6)
        
        for i, model in enumerate(df["model_name"]):
            ax5.annotate(model, (df["training_time"].iloc[i], df["gpu_memory_usage"].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        ax5.set_xlabel("Training Time (hours)")
        ax5.set_ylabel("GPU Memory Usage (GB)")
        ax5.set_title("Resource Efficiency (bubble size = parameters)")
        
        # 6. Scoring Strategy Comparison
        ax6 = axes[1, 2]
        strategies = ["composite_score", "quality_focused", "performance_focused", "efficiency_focused"]
        strategy_data = df[strategies].values
        
        im = ax6.imshow(strategy_data.T, cmap='RdYlGn', aspect='auto')
        ax6.set_xticks(range(len(df)))
        ax6.set_xticklabels(df["model_name"], rotation=45)
        ax6.set_yticks(range(len(strategies)))
        ax6.set_yticklabels(strategies)
        ax6.set_title("Scoring Strategy Comparison")
        
        # Add colorbar
        plt.colorbar(im, ax=ax6, label='Score')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.comparison_dir / "phase3_model_comparison.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved: {viz_file}")
        
        return viz_file
    
    def generate_executive_summary(self, rankings: List[ModelRanking], df: pd.DataFrame) -> str:
        """Generate executive summary for stakeholders"""
        
        winner = rankings[0]
        
        summary = f"""# 🏆 Phase 3 Executive Summary
        
**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Models Evaluated**: {len(rankings)}
**Evaluation Scope**: Quality Assessment, WAF Bypass Testing, Resource Analysis

## 🥇 Winning Model: {winner.model_name}

**Overall Score**: {winner.composite_score:.3f}/1.0
**Key Performance Indicators**:
- Quality Score: {df[df['model_name'] == winner.model_name]['quality_score'].iloc[0]:.3f}
- WAF Bypass Rate: {df[df['model_name'] == winner.model_name]['waf_bypass_rate'].iloc[0]:.3f}
- Training Time: {df[df['model_name'] == winner.model_name]['training_time'].iloc[0]:.1f} hours

### Strengths:
"""
        
        for strength in winner.strengths:
            summary += f"- ✅ {strength}\n"
        
        if winner.weaknesses:
            summary += "\n### Areas for Improvement:\n"
            for weakness in winner.weaknesses:
                summary += f"- ⚠️ {weakness}\n"
        
        summary += f"\n**Recommendation**: {winner.recommendation}\n\n"
        
        # Success criteria analysis
        best_quality = df["quality_score"].max()
        best_waf = df["waf_bypass_rate"].max()
        min_training = df["training_time"].min()
        
        summary += "## 📊 Success Criteria Status\n\n"
        summary += f"- Quality Target (≥0.55): {'✅' if best_quality >= 0.55 else '❌'} {best_quality:.3f}\n"
        summary += f"- WAF Bypass Target (≥0.70): {'✅' if best_waf >= 0.70 else '❌'} {best_waf:.3f}\n"
        summary += f"- Training Efficiency (≤10h): {'✅' if min_training <= 10.0 else '❌'} {min_training:.1f}h\n"
        
        # Model rankings
        summary += "\n## 🏅 Complete Rankings\n\n"
        for ranking in rankings:
            summary += f"**{ranking.rank}. {ranking.model_name}** - {ranking.composite_score:.3f}\n"
            if ranking.rank <= 2:
                summary += f"   - {ranking.recommendation}\n"
        
        # Performance improvements
        phase2_quality = 0.463
        phase2_waf = 0.60
        
        quality_improvement = ((best_quality / phase2_quality - 1) * 100)
        waf_improvement = ((best_waf / phase2_waf - 1) * 100)
        
        summary += f"\n## 🚀 Phase 2 vs Phase 3 Improvements\n\n"
        summary += f"- Quality: +{quality_improvement:.1f}% improvement\n"
        summary += f"- WAF Bypass: +{waf_improvement:.1f}% improvement\n"
        
        return summary
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run complete comparison analysis"""
        print("🔍 Loading Phase 3 evaluation results...")
        
        evaluations = self.load_evaluations()
        if not evaluations:
            print("❌ No evaluation results found")
            return {}
        
        print(f"📊 Analyzing {len(evaluations)} models...")
        
        # Create DataFrame
        df = pd.DataFrame(evaluations)
        
        # Calculate composite scores
        df = self.calculate_composite_scores(df)
        
        # Create detailed rankings
        rankings = self.create_detailed_rankings(df)
        
        # Generate comparison matrix
        comparison_matrix = self.generate_comparison_matrix(df)
        
        # Create visualizations
        viz_file = self.create_visualizations(df, rankings)
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(rankings, df)
        
        # Save detailed results
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_rankings": [
                {
                    "rank": r.rank,
                    "model_name": r.model_name,
                    "composite_score": r.composite_score,
                    "strengths": r.strengths,
                    "weaknesses": r.weaknesses,
                    "recommendation": r.recommendation
                } for r in rankings
            ],
            "comparison_weights": {
                "quality": self.weights.quality,
                "waf_bypass": self.weights.waf_bypass,
                "efficiency": self.weights.efficiency,
                "resource": self.weights.resource
            },
            "performance_summary": {
                "best_quality": float(df["quality_score"].max()),
                "best_waf_bypass": float(df["waf_bypass_rate"].max()),
                "fastest_training": float(df["training_time"].min()),
                "most_efficient": float(df["gpu_memory_usage"].min())
            }
        }
        
        # Save files
        results_file = self.comparison_dir / "detailed_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_file = self.comparison_dir / "executive_summary.md"
        with open(summary_file, 'w') as f:
            f.write(executive_summary)
        
        matrix_file = self.comparison_dir / "comparison_matrix.csv"
        comparison_matrix.to_csv(matrix_file)
        
        detailed_file = self.comparison_dir / "detailed_model_data.csv"
        df.to_csv(detailed_file, index=False)
        
        print(f"\n🎯 Comprehensive comparison complete!")
        print(f"🏆 Winner: {rankings[0].model_name} (Score: {rankings[0].composite_score:.3f})")
        print(f"📁 Results saved in: {self.comparison_dir}")
        print(f"📋 Executive summary: {summary_file}")
        
        return results

def main():
    """Main comparison pipeline"""
    
    # Default balanced weighting
    weights = ComparisonWeights(
        quality=0.40,
        waf_bypass=0.35,
        efficiency=0.15,
        resource=0.10
    )
    
    comparator = Phase3ModelComparison(weights)
    results = comparator.run_comprehensive_comparison()
    
    if results:
        print("\n✅ Model comparison analysis complete!")
    else:
        print("\n❌ Comparison analysis failed!")

if __name__ == "__main__":
    main()