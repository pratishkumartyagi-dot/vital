export class CameraCalibration {
  private static readonly CALIBRATION_DATABASE = [
    { model: 'Pixel 4', manufacturer: 'Google', zlo: 22.5, sensorLinearSlope: 1.0 },
    { model: 'Pixel 7', manufacturer: 'Google', zlo: 14.0, sensorLinearSlope: 1.02 },
    { model: 'Pixel 7 Pro', manufacturer: 'Google', zlo: 14.0, sensorLinearSlope: 1.02 },
    { model: 'Pixel 7a', manufacturer: 'Google', zlo: 14.0, sensorLinearSlope: 1.02 },
    { model: 'Pixel 8', manufacturer: 'Google', zlo: 14.0, sensorLinearSlope: 1.02 },
    { model: 'Pixel 8 Pro', manufacturer: 'Google', zlo: 14.0, sensorLinearSlope: 1.02 },
    { model: 'Galaxy S22', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S22+', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S22 Ultra', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S23', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S23+', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S23 Ultra', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S24', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S24+', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Galaxy S24 Ultra', manufacturer: 'Samsung', zlo: 19.6, sensorLinearSlope: 0.98 },
    { model: 'Moto G 2022', manufacturer: 'Motorola', zlo: 14.9, sensorLinearSlope: 1.0 },
    { model: 'Moto G Power 2022', manufacturer: 'Motorola', zlo: 14.9, sensorLinearSlope: 1.0 },
    { model: 'Moto G Stylus 2022', manufacturer: 'Motorola', zlo: 14.9, sensorLinearSlope: 1.0 },
    { model: 'OnePlus 10 Pro', manufacturer: 'OnePlus', zlo: 18.0, sensorLinearSlope: 1.0 },
    { model: 'OnePlus 11', manufacturer: 'OnePlus', zlo: 18.0, sensorLinearSlope: 1.0 },
    { model: 'Xiaomi 12', manufacturer: 'Xiaomi', zlo: 16.5, sensorLinearSlope: 1.0 },
    { model: 'Xiaomi 13', manufacturer: 'Xiaomi', zlo: 16.5, sensorLinearSlope: 1.0 },
    { model: 'iPhone 13', manufacturer: 'Apple', zlo: 12.0, sensorLinearSlope: 1.05 },
    { model: 'iPhone 13 Pro', manufacturer: 'Apple', zlo: 12.0, sensorLinearSlope: 1.05 },
    { model: 'iPhone 14', manufacturer: 'Apple', zlo: 12.0, sensorLinearSlope: 1.05 },
    { model: 'iPhone 14 Pro', manufacturer: 'Apple', zlo: 12.0, sensorLinearSlope: 1.05 },
    { model: 'iPhone 15', manufacturer: 'Apple', zlo: 12.0, sensorLinearSlope: 1.05 },
    { model: 'iPhone 15 Pro', manufacturer: 'Apple', zlo: 12.0, sensorLinearSlope: 1.05 },
  ];

  private static readonly DEFAULT_ZLO = 16.0;
  private static readonly DEFAULT_SLOPE = 1.0;

  static getCalibrationInfo(model: string, manufacturer: string, apiLevel: number): {
    zlo: number;
    sensorLinearSlope: number;
    model: string;
    manufacturer: string;
    matched: boolean;
    profile: any;
  } {
    for (const profile of this.CALIBRATION_DATABASE) {
      if (profile.model.toLowerCase() === model.toLowerCase() &&
          profile.manufacturer.toLowerCase() === manufacturer.toLowerCase()) {
        return {
          zlo: profile.zlo,
          sensorLinearSlope: profile.sensorLinearSlope,
          model: profile.model,
          manufacturer: profile.manufacturer,
          matched: true,
          profile: profile,
        };
      }
    }
    return {
      zlo: this.DEFAULT_ZLO,
      sensorLinearSlope: this.DEFAULT_SLOPE,
      model: model,
      manufacturer: manufacturer,
      matched: false,
      profile: null,
    };
  }
}