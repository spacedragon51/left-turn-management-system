"""
Data Analyzer Module - Generates insights and recommendations from parsed data
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Analyze traffic data and generate actionable insights
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize analyzer with parsed data
        
        Args:
            data: Parsed data from PDFParser
        """
        self.data = data
        self.violations_df = pd.DataFrame(data.get('violations', []))
        self.volume_df = pd.DataFrame(data.get('traffic_volume', []))
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete analysis
        
        Returns:
            Complete analysis report
        """
        violations_analysis = self._analyze_violations()
        risk_assessment = self._calculate_risk(violations_analysis)
        recommendations = self._generate_recommendations(violations_analysis, risk_assessment)
        
        return {
            'violation_analysis': violations_analysis,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'statistics': self._get_statistics()
        }
    
    def _analyze_violations(self) -> Dict[str, Any]:
        """Analyze violation patterns"""
        if self.violations_df.empty:
            return {'total_violations': 0, 'peak_hours': {}}
        
        analysis = {
            'total_violations': len(self.violations_df),
            'peak_hours': {}
        }
        
        # Try to extract hour from timestamp if available
        time_cols = [c for c in self.violations_df.columns if 'time' in c.lower() or 'hour' in c.lower()]
        if time_cols:
            try:
                times = self.violations_df[time_cols[0]].astype(str)
                hours = []
                for t in times:
                    # Try to extract hour from various formats
                    if ':' in t:
                        hour_part = t.split(':')[0]
                        try:
                            hours.append(int(hour_part))
                        except ValueError:
                            pass
                
                if hours:
                    hour_counts = pd.Series(hours).value_counts()
                    analysis['peak_hours'] = hour_counts.head(3).to_dict()
            except Exception as e:
                logger.warning(f"Could not extract hours: {e}")
        
        # Vehicle type distribution
        type_cols = [c for c in self.violations_df.columns if 'type' in c.lower() or 'class' in c.lower() or 'vehicle' in c.lower()]
        if type_cols:
            try:
                analysis['vehicle_distribution'] = self.violations_df[type_cols[0]].value_counts().to_dict()
            except Exception:
                pass
        
        return analysis
    
    def _calculate_risk(self, violations: Dict) -> Dict[str, Any]:
        """Calculate risk score"""
        total = violations.get('total_violations', 0)
        peak_hours = violations.get('peak_hours', {})
        
        risk_score = 0
        
        # Violation volume
        if total > 100:
            risk_score += 40
        elif total > 50:
            risk_score += 25
        elif total > 10:
            risk_score += 10
        
        # Peak concentration
        if peak_hours:
            max_peak = max(peak_hours.values()) if peak_hours else 0
            if max_peak > 20:
                risk_score += 30
            elif max_peak > 10:
                risk_score += 15
        
        # Determine level
        risk_score = min(risk_score, 100)
        if risk_score > 60:
            level = 'HIGH'
        elif risk_score > 30:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {
            'score': risk_score,
            'level': level,
            'factors': {
                'violation_volume': total,
                'peak_concentration': max(peak_hours.values()) if peak_hours else 0
            }
        }
    
    def _generate_recommendations(self, violations: Dict, risk: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        risk_level = risk.get('level', 'MEDIUM')
        total_violations = violations.get('total_violations', 0)
        
        # Risk-based recommendations
        if risk_level == 'HIGH':
            recommendations.append("🚨 IMMEDIATE: Implement protected left-turn signal during peak hours")
            recommendations.append("📹 Deploy AI-based enforcement cameras at this intersection")
            recommendations.append("🚧 Install physical channelizers to separate free-left lane")
        elif risk_level == 'MEDIUM':
            recommendations.append("⚠️ Schedule periodic protected left-turn interventions")
            recommendations.append("📊 Monitor violation patterns for 2 weeks before permanent changes")
        elif total_violations < 10:
            recommendations.append("✅ Current operations are within safe limits. Continue monitoring.")
        
        # Peak hour recommendations
        peak_hours = violations.get('peak_hours', {})
        if peak_hours:
            peak_times = ', '.join([f"{h}:00" for h in peak_hours.keys()])
            recommendations.append(f"🕐 Deploy traffic marshals during peak hours: {peak_times}")
        
        # Vehicle-specific
        vehicle_dist = violations.get('vehicle_distribution', {})
        if vehicle_dist:
            if any('bus' in str(v).lower() for v in vehicle_dist.keys()):
                recommendations.append("🚌 Create dedicated bus bay with clear lane separation")
            if any('motor' in str(v).lower() or 'bike' in str(v).lower() for v in vehicle_dist.keys()):
                recommendations.append("🛵 Implement dedicated two-wheeler waiting zone")
        
        # Pedestrian safety
        if self.data.get('pedestrian_data'):
            recommendations.append("🚶 Add pedestrian crossing signals with countdown timers")
        
        if not recommendations:
            recommendations.append("✅ Continue regular monitoring and enforcement")
        
        return recommendations
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics"""
        return {
            'violation_count': len(self.violations_df),
            'traffic_volume_records': len(self.volume_df),
            'pedestrian_records': len(self.data.get('pedestrian_data', []))
        }